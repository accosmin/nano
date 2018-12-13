#include "model_gboost.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "gboost_loss_avg.h"
#include "gboost_loss_var.h"

using namespace nano;

template <typename tweak_learner>
void model_gboost_t<tweak_learner>::to_json(json_t& json) const
{
        nano::to_json(json,
                "rounds", m_rounds,
                "patience", m_patience,
                "solver", m_solver,
                "cumloss", to_string(m_cumloss) + join(enum_values<cumloss>()),
                "shrinkage", to_string(m_shrinkage) + join(enum_values<shrinkage>()),
                "subsampling", to_string(m_subsampling) + join(enum_values<subsampling>()));
}

template <typename tweak_learner>
void model_gboost_t<tweak_learner>::from_json(const json_t& json)
{
        nano::from_json(json,
                "rounds", m_rounds,
                "patience", m_patience,
                "solver", m_solver,
                "cumloss", m_cumloss,
                "shrinkage", m_shrinkage,
                "subsampling", m_subsampling);
}

template <typename tweak_learner>
training_t model_gboost_t<tweak_learner>::train(const task_t& task, const size_t fold, const loss_t& loss)
{
        m_idims = task.idims();
        m_odims = task.odims();

        tuner_t tuner;

        switch (m_shrinkage)
        {
        case shrinkage::off:
                tuner.add_finite("shrinkage", 1.0);
                break;

        case shrinkage::on:
                tuner.add_finite("shrinkage", 0.1, 0.2, 0.5, 1.0);
                break;
        }

        switch (m_subsampling)
        {
        case subsampling::off:
                tuner.add_finite("subsampling", 100);
                break;

        case subsampling::on:
                tuner.add_finite("subsampling", 10, 20, 50, 100);
                break;
        }

        switch (m_cumloss)
        {
        case cumloss::variance:
                tuner.add_pow10s("lambda", 0.0, -6, +6);
                return train<gboost_loss_var_t<tweak_learner>>(task, fold, loss, tuner);

        case cumloss::average:
        default:
                tuner.add_finite("lambda", 1.0);
                return train<gboost_loss_avg_t<tweak_learner>>(task, fold, loss, tuner);
        }
}

template <typename tweak_learner>
tensor3d_t model_gboost_t<tweak_learner>::output(const tensor3d_t& input) const
{
        assert(input.dims() == m_idims);

        tensor3d_t output(m_odims);
        output.zero();

        for (const auto& wlearner : m_wlearners)
        {
                output.vector() += wlearner.output(input).vector();
        }

        return output;
}

template <typename tweak_learner>
bool model_gboost_t<tweak_learner>::save(obstream_t& stream) const
{
        if (    !stream.write(m_idims) ||
                !stream.write(m_odims) ||
                !stream.write(m_rounds) ||
                !stream.write(m_cumloss) ||
                !stream.write(m_shrinkage) ||
                !stream.write(m_subsampling) ||
                !stream.write(m_wlearners.size()))
        {
                return false;
        }

        for (const auto& wlearner : m_wlearners)
        {
                if (!wlearner.save(stream))
                {
                        return false;
                }
        }

        return true;
}

template <typename tweak_learner>
bool model_gboost_t<tweak_learner>::load(ibstream_t& stream)
{
        size_t n_wlearners = 0;
        if (    !stream.read(m_idims) ||
                !stream.read(m_odims) ||
                !stream.read(m_rounds) ||
                !stream.read(m_cumloss) ||
                !stream.read(m_shrinkage) ||
                !stream.read(m_subsampling) ||
                !stream.read(n_wlearners))
        {
                return false;
        }

        m_wlearners.resize(n_wlearners);
        for (auto& wlearner : m_wlearners)
        {
                if (!wlearner.load(stream))
                {
                        return false;
                }
        }

        // todo: more verbose loading (#weak learners, feature or coefficient statistics, idims...)

        return true;
}

template class nano::model_gboost_t<nano::wlearner_linear_t>;
template class nano::model_gboost_t<nano::wlearner_real_stump_t>;
template class nano::model_gboost_t<nano::wlearner_discrete_stump_t>;
