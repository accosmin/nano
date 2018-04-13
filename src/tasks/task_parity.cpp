#include "task_parity.h"

using namespace nano;

parity_task_t::parity_task_t() :
        mem_tensor_task_t(make_dims(32, 1, 1), make_dims(1, 1, 1), 1)
{
}

void parity_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "n", m_dims, "count", m_count);
        reconfig(make_dims(m_dims, 1, 1), make_dims(1, 1, 1), 1);
}

void parity_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "n", m_dims, "count", m_count);
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
                size_t ones = 0;
                for (tensor_size_t x = 0; x < m_dims; ++ x)
                {
                        if ((udist_bit(rng) & 0x01))
                        {
                                bitset(x) = 1;
                                ++ ones;
                        }
                        else
                        {
                                bitset(x) = 0;
                        }
                }

                const auto hash = i;
                const auto label = (ones % 2) ? "odd" : "even";
                const auto target = class_target(1 - (ones % 2), 1);

                add_chunk(bitset, hash);
                add_sample(make_fold(0), i, target, label);
        }

        return true;
}
