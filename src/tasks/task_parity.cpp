#include "task_parity.h"

using namespace nano;

parity_task_t::parity_task_t() :
        mem_tensor_task_t(make_dims(32, 1, 1), make_dims(1, 1, 1), 10)
{
}

void parity_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "n", m_dims, "count", m_count, "folds", m_folds);
        reconfig(make_dims(m_dims, 1, 1), make_dims(1, 1, 1), m_folds);
}

void parity_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "n", m_dims, "count", m_count, "folds", m_folds);
}

bool parity_task_t::populate()
{
        auto rng = make_rng();
        auto udist_bit = make_udist<tensor_size_t>(1, 1024);

        tensor3d_t bitset(m_dims, 1, 1);

        // generate samples
        reserve_chunks(m_count);
        for (size_t i = 0; i < m_count; ++ i)
        {
                for (tensor_size_t x = 0; x < m_dims; ++ x)
                {
                        if ((udist_bit(rng) & 0x01))
                        {
                                bitset(x) = 1;
                        }
                        else
                        {
                                bitset(x) = 0;
                        }
                }

                const auto hash = i;
                add_chunk(bitset, hash);
        }

        // generate folds
        for (size_t f = 0; f < m_folds; ++ f)
        {
                const auto protocols = split3(m_count, protocol::train, 40, protocol::valid, 30, protocol::test);

                for (size_t i = 0; i < m_count; ++ i)
                {
                        const size_t ones = (chunk(i).array() > scalar_t(0.5)).count();

                        const auto label = (ones % 2) ? "odd" : "even";
                        const auto target = class_target(1, 1 - (ones % 2));
                        const auto fold = fold_t{f, protocols[i]};

                        add_sample(fold, i, target, label);
                }
        }

        return true;
}
