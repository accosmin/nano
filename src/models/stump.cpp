#include "task.h"
#include "stump.h"
#include "core/tpool.h"
#include "core/ibstream.h"
#include "core/obstream.h"

using namespace nano;

void stump_t::fit(const task_t& task, const fold_t& fold, const tensor4d_t& residuals, const wlearner_type type)
{
        assert(cat_dims(task.size(fold), task.odims()) == residuals.dims());

        const auto& tpool = tpool_t::instance();

        stumps_t tstumps(tpool.workers());
        scalars_t tvalues(tpool.workers(), std::numeric_limits<scalar_t>::max());

        loopit(nano::size(task.idims()), [&] (const tensor_size_t feature, const size_t t)
        {
                const auto fvalues = stump_t::fvalues(task, fold, feature);
                const auto thresholds = stump_t::thresholds(fvalues);

                stump_t stump;
                const auto value = stump.fit(task, residuals, feature, fvalues, thresholds, type);
                if (value < tvalues[t])
                {
                        tvalues[t] = value;
                        std::swap(tstumps[t], stump);
                }
        });

        *this = tstumps[std::min_element(tvalues.begin(), tvalues.end()) - tvalues.begin()];
}

scalar_t stump_t::fit(const task_t& task, const tensor4d_t& residuals,
        const tensor_size_t feature, const scalars_t& fvalues, const scalars_t& thresholds,
        const wlearner_type type)
{
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
                        m_feature = feature;
                        m_threshold = threshold;
                        m_outputs.resize(cat_dims(2, task.odims()));

                        switch (type)
                        {
                        case wlearner_type::discrete:
                                m_outputs.vector(0) = residuals_neg1.array().sign();
                                m_outputs.vector(1) = residuals_pos1.array().sign();
                                break;

                        default:
                                m_outputs.vector(0) = residuals_neg1.vector() / cnt_neg;
                                m_outputs.vector(1) = residuals_pos1.vector() / cnt_pos;
                                break;
                        }

                        // todo: implement subsampling
                        // todo: implement fitting stumps on training loss and evaluating then on the validation error
                }
        }

        return value;
}

scalars_t stump_t::fvalues(const task_t& task, const fold_t& fold, const tensor_size_t feature)
{
        scalars_t fvalues(task.size(fold));
        for (size_t i = 0, size = task.size(fold); i < size; ++ i)
        {
                const auto input = task.input(fold, i);
                fvalues[i] = input(feature);
        }

        return fvalues;
}

scalars_t stump_t::thresholds(const scalars_t& fvalues)
{
        auto thresholds = fvalues;
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(
                std::unique(thresholds.begin(), thresholds.end()),
                thresholds.end());

        return thresholds;
}

bool stump_t::save(obstream_t& stream) const
{
        return  stream.write(m_feature) &&
                stream.write(m_threshold) &&
                stream.write_tensor(m_outputs);
}

bool stump_t::load(ibstream_t& stream)
{
        return  stream.read(m_feature) &&
                stream.read(m_threshold) &&
                stream.read_tensor(m_outputs);
}
