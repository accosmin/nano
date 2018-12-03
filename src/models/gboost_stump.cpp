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
                "eval", to_string(m_weval) + join(enum_values<wlearner_eval>()),
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
                "eval", m_weval,
                "cumloss", m_cumloss,
                "shrinkage", m_shrinkage,
                "subsampling", m_subsampling);
}

trainer_result_t gboost_stump_t::train(const task_t& task, const size_t fold, const loss_t& loss)
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

std::pair<scalar_t, stump_t> gboost_stump_t::fit(const task_t& task, const tensor4d_t& residuals,
        const tensor_size_t feature, const scalars_t& fvalues, const scalars_t& thresholds) const
{
        stump_t stump;
        scalar_t value = std::numeric_limits<scalar_t>::max();

        tensor3d_t residuals_pos1(task.odims()), residuals_pos2(task.odims());
        tensor3d_t residuals_neg1(task.odims()), residuals_neg2(task.odims());

        for (size_t t = 0; t + 1 < thresholds.size(); ++ t)
        {
                const auto threshold = (thresholds[t + 0] + thresholds[t + 1]) / 2;

                residuals_pos1.zero(), residuals_pos2.zero();
                residuals_neg1.zero(), residuals_neg2.zero();

                int cnt_pos = 0, cnt_neg = 0;
                for (size_t i = 0, size = fvalues.size(); i < size; ++ i)
                {
                        const auto residual = residuals.array(i);
                        if (fvalues[i] < threshold)
                        {
                                cnt_neg ++;
                                residuals_neg1.array() += residual;
                                residuals_neg2.array() += residual * residual;
                        }
                        else
                        {
                                cnt_pos ++;
                                residuals_pos1.array() += residual;
                                residuals_pos2.array() += residual * residual;
                        }
                }

                const auto tvalue =
                        (residuals_pos2.array() - residuals_pos1.array().square() / cnt_pos).sum() +
                        (residuals_neg2.array() - residuals_neg1.array().square() / cnt_neg).sum();

                if (tvalue < value)
                {
                        value = tvalue;
                        stump.m_feature = feature;
                        stump.m_threshold = threshold;
                        stump.m_outputs.resize(cat_dims(2, task.odims()));
                        switch (m_wtype)
                        {
                        case wlearner_type::discrete:
                                stump.m_outputs.vector(0) = residuals_neg1.array().sign();
                                stump.m_outputs.vector(1) = residuals_pos1.array().sign();
                                break;

                        default:
                                stump.m_outputs.vector(0) = residuals_neg1.vector() / cnt_neg;
                                stump.m_outputs.vector(1) = residuals_pos1.vector() / cnt_pos;
                                break;
                        }

                        // todo: implement subsampling
                        // todo: implement fitting stumps on training loss and evaluating then on the validation error
                }

                // todo: move this function to stump_t (need to have something similar for other weak learners as well)
        }

        return std::make_pair(value, stump);
}

tensor3d_t gboost_stump_t::output(const tensor3d_t& input) const
{
        assert(input.dims() == m_idims);

        tensor3d_t output(m_odims);
        output.zero();

        const auto idata = input.array();
        auto odata = output.array();

        for (const auto& stump : m_stumps)
        {
                const auto oindex = idata(stump.m_feature) < stump.m_threshold ? 0 : 1;
                odata.array() += stump.m_outputs.array(oindex);
        }

        return output;
}

bool gboost_stump_t::save(obstream_t& stream) const
{
        if (    !stream.write(m_idims) ||
                !stream.write(m_odims) ||
                !stream.write(m_rounds) ||
                !stream.write(m_wtype) ||
                !stream.write(m_weval) ||
                !stream.write(m_cumloss) ||
                !stream.write(m_shrinkage) ||
                !stream.write(m_subsampling) ||
                !stream.write(m_stumps.size()))
        {
                return false;
        }

        for (const auto& stump : m_stumps)
        {
                assert(stump.m_feature >= 0 && stump.m_feature < nano::size(m_idims));
                assert(stump.m_outputs.dims() == cat_dims(2, m_odims));

                if (    !stream.write(stump.m_feature) ||
                        !stream.write(stump.m_threshold) ||
                        !stream.write_tensor(stump.m_outputs))
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
                !stream.read(m_weval) ||
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
                if (    !stream.read(stump.m_feature) ||
                        !stream.read(stump.m_threshold) ||
                        !stream.read_tensor(stump.m_outputs) ||
                        stump.m_feature < 0 ||
                        stump.m_feature >= nano::size(m_idims) ||
                        stump.m_outputs.dims() != cat_dims(2, m_odims))
                {
                        return false;
                }
        }

        // todo: more verbose loading (#stumps, feature or coefficient statistics, idims...)

        return true;
}

probes_t gboost_stump_t::probes() const
{
        // todo: add probes here to measure the training and the evaluation time
        probes_t probes;
        return probes;
}
