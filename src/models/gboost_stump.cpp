#include "gboost_stump.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "gboost_loss_avg.h"
#include "gboost_loss_var.h"

using namespace nano;

void gboost_stump_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "rounds", m_rounds,
                "patience", m_patience,
                "solver", m_solver,
                "type", to_string(m_wtype) + join(enum_values<wlearner_type>()),
                "cumloss", to_string(m_cumloss) + join(enum_values<cumloss>()),
                "shrinkage", to_string(m_shrinkage) + join(enum_values<shrinkage>()),
                "subsampling", to_string(m_subsampling) + join(enum_values<subsampling>()));
}

void gboost_stump_t::from_json(const json_t& json)
{
        nano::from_json(json,
                "rounds", m_rounds,
                "patience", m_patience,
                "solver", m_solver,
                "type", m_wtype,
                "cumloss", m_cumloss,
                "shrinkage", m_shrinkage,
                "subsampling", m_subsampling);
}

training_t gboost_stump_t::train(const task_t& task, const size_t fold, const loss_t& loss)
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
                return train<gboost_loss_var_t<stump_t>>(task, fold, loss, tuner);

        case cumloss::average:
        default:
                tuner.add_finite("lambda", 1.0);
                return train<gboost_loss_avg_t<stump_t>>(task, fold, loss, tuner);
        }
}

tensor3d_t gboost_stump_t::output(const tensor3d_t& input) const
{
        assert(input.dims() == m_idims);

        tensor3d_t output(m_odims);
        output.zero();

        for (const auto& stump : m_stumps)
        {
                output.vector() += stump.output(input).vector();
        }

        return output;
}

bool gboost_stump_t::save(obstream_t& stream) const
{
        if (    !stream.write(m_idims) ||
                !stream.write(m_odims) ||
                !stream.write(m_rounds) ||
                !stream.write(m_wtype) ||
                !stream.write(m_cumloss) ||
                !stream.write(m_shrinkage) ||
                !stream.write(m_subsampling) ||
                !stream.write(m_stumps.size()))
        {
                return false;
        }

        for (const auto& stump : m_stumps)
        {
                assert(stump.feature() >= 0 && stump.feature() < nano::size(m_idims));
                assert(stump.outputs().dims() == cat_dims(2, m_odims));

                if (!stump.save(stream))
                {
                        return false;
                }
        }

        return true;
}

bool gboost_stump_t::load(ibstream_t& stream)
{
        size_t n_stumps = 0;
        if (    !stream.read(m_idims) ||
                !stream.read(m_odims) ||
                !stream.read(m_rounds) ||
                !stream.read(m_wtype) ||
                !stream.read(m_cumloss) ||
                !stream.read(m_shrinkage) ||
                !stream.read(m_subsampling) ||
                !stream.read(n_stumps))
        {
                return false;
        }

        m_stumps.resize(n_stumps);
        for (auto& stump : m_stumps)
        {
                if (    !stump.load(stream) ||
                        stump.feature() < 0 ||
                        stump.feature() >= nano::size(m_idims) ||
                        stump.outputs().dims() != cat_dims(2, m_odims))
                {
                        return false;
                }
        }

        // todo: more verbose loading (#stumps, feature or coefficient statistics, idims...)

        return true;
}
