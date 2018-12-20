#include "version.h"
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
scalar_t wlearner_stump_t<type>::fit(const task_t& task, const fold_t& fold, const tensor4d_t& gradients,
        const indices_t& indices, const tensor_size_t feature)
{
        const auto fvalues = get_fvalues(task, fold, feature);
        const auto thresholds = get_thresholds(fvalues);

        m_outputs.resize(cat_dims(2, task.odims()));

        scalar_t value = std::numeric_limits<scalar_t>::max();
        tensor3d_t res_pos1(task.odims()), res_pos2(task.odims());
        tensor3d_t res_neg1(task.odims()), res_neg2(task.odims());

        for (size_t t = 0; t + 1 < thresholds.size(); ++ t)
        {
                const auto threshold = (thresholds[t + 0] + thresholds[t + 1]) / 2;

                res_pos1.zero(), res_pos2.zero();
                res_neg1.zero(), res_neg2.zero();

                int cnt_pos = 0, cnt_neg = 0;
                for (const auto i : indices)
                {
                        assert(i < task.size(fold));
                        assert(i < static_cast<size_t>(gradients.size<0>()));

                        const auto gradient = gradients.array(i);
                        if (fvalues[i] < threshold)
                        {
                                cnt_neg ++;
                                res_neg1.array() -= gradient;
                                res_neg2.array() += gradient * gradient;
                        }
                        else
                        {
                                cnt_pos ++;
                                res_pos1.array() -= gradient;
                                res_pos2.array() += gradient * gradient;
                        }
                }

                switch (type)
                {
                case stump_type::discrete:
                        try_fit(cnt_neg, res_neg1, res_neg2, res_neg1.array().sign(),
                                cnt_pos, res_pos1, res_pos2, res_pos1.array().sign(),
                                feature, threshold, value);
                        break;

                case stump_type::real:
                default:
                        try_fit(cnt_neg, res_neg1, res_neg2, res_neg1.array() / cnt_neg,
                                cnt_pos, res_pos1, res_pos2, res_pos1.array() / cnt_pos,
                                feature, threshold, value);
                        break;
                }
        }

        return value;
}

template <stump_type type>
bool wlearner_stump_t<type>::save(obstream_t& stream) const
{
        const auto vmajor = static_cast<uint8_t>(major_version);
        const auto vminor = static_cast<uint8_t>(minor_version);

        return  stream.write(vmajor) &&
                stream.write(vminor) &&
                stream.write(type) &&
                stream.write(m_feature) &&
                stream.write(m_threshold) &&
                stream.write_tensor(m_outputs);
}

template <stump_type type>
bool wlearner_stump_t<type>::load(ibstream_t& stream)
{
        uint8_t vmajor = 0x00;
        uint8_t vminor = 0x00;

        stump_type other_type;
        return  stream.read(vmajor) &&
                stream.read(vminor) &&
                stream.read(other_type) &&
                stream.read(m_feature) &&
                stream.read(m_threshold) &&
                stream.read_tensor(m_outputs) &&
                type == other_type;
}

template class NANO_PUBLIC nano::wlearner_stump_t<stump_type::real>;
template class NANO_PUBLIC nano::wlearner_stump_t<stump_type::discrete>;
