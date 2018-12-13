#include "task.h"
#include "core/tpool.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "wlearner_stump.h"

using namespace nano;

static auto get_fvalues(const task_t& task, const fold_t& fold, const tensor_size_t feature)
{
        scalars_t fvalues(task.size(fold));
        for (size_t i = 0, size = task.size(fold); i < size; ++ i)
        {
                const auto input = task.input(fold, i);
                fvalues[i] = input(feature);
        }

        return fvalues;
}

static auto get_thresholds(const scalars_t& fvalues)
{
        auto thresholds = fvalues;
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(
                std::unique(thresholds.begin(), thresholds.end()),
                thresholds.end());

        return thresholds;
}

template <stump_type type>
void wlearner_stump_t<type>::fit(const task_t& task, const fold_t& fold, const tensor4d_t& gradients)
{
        assert(cat_dims(task.size(fold), task.odims()) == gradients.dims());

        const auto& tpool = tpool_t::instance();

        std::vector<wlearner_stump_t> learners(tpool.workers());
        scalars_t tvalues(tpool.workers(), std::numeric_limits<scalar_t>::max());

        loopit(nano::size(task.idims()), [&] (const tensor_size_t feature, const size_t t)
        {
                wlearner_stump_t learner;
                const auto value = learner.fit(task, fold, gradients, feature);
                if (value < tvalues[t])
                {
                        tvalues[t] = value;
                        std::swap(learners[t], learner);
                }
        });

        *this = learners[std::min_element(tvalues.begin(), tvalues.end()) - tvalues.begin()];
}

template <stump_type type>
scalar_t wlearner_stump_t<type>::fit(const task_t& task, const fold_t& fold, const tensor4d_t& gradients,
        const tensor_size_t feature)
{
        const auto fvalues = get_fvalues(task, fold, feature);
        const auto thresholds = get_thresholds(fvalues);

        m_outputs.resize(cat_dims(2, task.odims()));

        scalar_t value = std::numeric_limits<scalar_t>::max();
        tensor3d_t gradients_pos1(task.odims()), gradients_pos2(task.odims());
        tensor3d_t gradients_neg1(task.odims()), gradients_neg2(task.odims());

        for (size_t t = 0; t + 1 < thresholds.size(); ++ t)
        {
                const auto threshold = (thresholds[t + 0] + thresholds[t + 1]) / 2;

                gradients_pos1.zero(), gradients_pos2.zero();
                gradients_neg1.zero(), gradients_neg2.zero();

                int cnt_pos = 0, cnt_neg = 0;
                for (size_t i = 0, size = fvalues.size(); i < size; ++ i)
                {
                        const auto gradient = gradients.array(i);
                        if (fvalues[i] < threshold)
                        {
                                cnt_neg ++;
                                gradients_neg1.array() -= gradient;
                                gradients_neg2.array() += gradient * gradient;
                        }
                        else
                        {
                                cnt_pos ++;
                                gradients_pos1.array() -= gradient;
                                gradients_pos2.array() += gradient * gradient;
                        }
                }

                switch (type)
                {
                case stump_type::discrete:
                        try_fit(cnt_neg, gradients_neg1, gradients_neg2, gradients_neg1.array().sign(),
                                cnt_pos, gradients_pos1, gradients_pos2, gradients_pos1.array().sign(),
                                feature, threshold, value);
                        break;

                case stump_type::real:
                default:
                        try_fit(cnt_neg, gradients_neg1, gradients_neg2, gradients_neg1.array() / cnt_neg,
                                cnt_pos, gradients_pos1, gradients_pos2, gradients_pos1.array() / cnt_pos,
                                feature, threshold, value);
                        break;
                }

                // todo: implement subsampling
        }

        return value;
}

template <stump_type type>
bool wlearner_stump_t<type>::save(obstream_t& stream) const
{
        return  stream.write(type) &&
                stream.write(m_feature) &&
                stream.write(m_threshold) &&
                stream.write_tensor(m_outputs);
}

template <stump_type type>
bool wlearner_stump_t<type>::load(ibstream_t& stream)
{
        stump_type other_type;
        return  stream.read(other_type) &&
                stream.read(m_feature) &&
                stream.read(m_threshold) &&
                stream.read_tensor(m_outputs) &&
                type == other_type;
}

template class nano::wlearner_stump_t<nano::stump_type::real>;
template class nano::wlearner_stump_t<nano::stump_type::discrete>;
