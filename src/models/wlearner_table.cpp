#include "version.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "wlearner_table.h"

using namespace nano;

template <table_type type>
scalar_t wlearner_table_t<type>::fit(const task_t& task, const fold_t& fold, const tensor4d_t& gradients,
        const indices_t& indices, const tensor_size_t feature)
{
        // todo: need to find the range of feature values
        // todo: may need an offset to obtain positive indices
        const auto n_fvalues = 42;

        m_outputs.resize(cat_dims(n_fvalues, task.odims()));

        tensor1d_t cnt(n_fvalues);
        tensor4d_t res1(cat_dims(n_fvalues, task.odims()));
        tensor4d_t res2(cat_dims(n_fvalues, task.odims()));

        cnt.zero();
        res1.zero();
        res2.zero();

        for (const auto i : indices)
        {
                assert(i < task.size(fold));
                assert(i < static_cast<size_t>(gradients.size<0>()));

                const auto input = task.input(fold, i);
                const auto gradient = gradients.array(i);
                const auto fvalue = static_cast<tensor_size_t>(input(feature));

                cnt(fvalue) ++;
                res1.array(fvalue) -= gradient;
                res2.array(fvalue) += gradient * gradient;
        }

        m_feature = feature;
        m_outputs.resize(cat_dims(n_fvalues, task.odims()));

        scalar_t value = 0;
        for (auto fv = 0; fv < n_fvalues; ++ fv)
        {
                switch (type)
                {
                case table_type::discrete:
                        m_outputs.array(fv) = res1.array(fv).sign();
                        break;

                case table_type::real:
                default:
                        m_outputs.array(fv) = res1.array(fv) / std::max(cnt(fv), scalar_t(1));
                        break;
                }

                value += (
                        + cnt(fv) * m_outputs.array(fv).square()
                        - 2 * m_outputs.array(fv) * res1.array(fv) +
                        res2.array(fv)).sum();
        }

        return value;
}

template <table_type type>
bool wlearner_table_t<type>::save(obstream_t& stream) const
{
        const auto vmajor = static_cast<uint8_t>(major_version);
        const auto vminor = static_cast<uint8_t>(minor_version);

        return  stream.write(vmajor) &&
                stream.write(vminor) &&
                stream.write(type) &&
                stream.write(m_feature) &&
                stream.write_tensor(m_outputs);
}

template <table_type type>
bool wlearner_table_t<type>::load(ibstream_t& stream)
{
        uint8_t vmajor = 0x00;
        uint8_t vminor = 0x00;

        table_type other_type;
        return  stream.read(vmajor) &&
                stream.read(vminor) &&
                stream.read(other_type) &&
                stream.read(m_feature) &&
                stream.read_tensor(m_outputs) &&
                type == other_type;
}

template class NANO_PUBLIC nano::wlearner_table_t<table_type::real>;
template class NANO_PUBLIC nano::wlearner_table_t<table_type::discrete>;
