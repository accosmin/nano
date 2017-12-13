#include "task_nparity.h"

#include <iostream>

using namespace nano;

nparity_task_t::nparity_task_t() :
        mem_tensor_task_t(make_dims(32, 1, 1), make_dims(1, 1, 1), 1)
{
}

json_reader_t& nparity_task_t::config(json_reader_t& reader)
{
        reader.object("n", m_dims, "count", m_count);
        reconfig(make_dims(m_dims, 1, 1), make_dims(1, 1, 1), 1);
        return reader;
}

json_writer_t& nparity_task_t::config(json_writer_t& writer) const
{
        return writer.object("n", m_dims, "count", m_count);
}

bool nparity_task_t::populate()
{
        auto rng_bit = make_rng<tensor_size_t>();

        tensor3d_t bitset(m_dims, 1, 1);

        reserve_chunks(m_count);

        // generate samples
        for (size_t i = 0; i < m_count; ++ i)
        {
                size_t ones = 0;
                for (tensor_size_t x = 0; x < m_dims; ++ x)
                {
                        if ((rng_bit() & 0x01))
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
                const auto target = class_target(ones % 2, 1);

                add_chunk(bitset, hash);
                add_sample(make_fold(0), i, target, label);
        }

        return true;
}
