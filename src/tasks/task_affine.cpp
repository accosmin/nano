#include "task_affine.h"
#include "tensor/numeric.h"

using namespace nano;

affine_task_t::affine_task_t() :
        mem_tensor_task_t(make_dims(32, 1, 1), make_dims(32, 1, 1), 1)
{
}

json_reader_t& affine_task_t::config(json_reader_t& reader)
{
        reader.object("isize", m_isize, "osize", m_osize, "noise", m_noise, "count", m_count, "type", m_type);
        reconfig(make_dims(m_isize, 1, 1), make_dims(m_osize, 1, 1), 1);
        return reader;
}

json_writer_t& affine_task_t::config(json_writer_t& writer) const
{
        return writer.object("isize", m_isize, "osize", m_osize, "noise", m_noise, "count", m_count,
                "type", m_type, "types", join(enum_values<affine_task_type>() ));
}

bool affine_task_t::populate()
{
        auto rng_noise = make_rng<scalar_t>(-m_noise, +m_noise);

        tensor2d_t weights(m_osize, m_isize);
        tensor1d_t bias(m_osize);
        tensor3d_t input(m_isize, 1, 1);
        tensor3d_t target(m_osize, 1, 1);

        weights.random();
        bias.random();

        // generate samples
        reserve_chunks(m_count);
        for (size_t i = 0; i < m_count; ++ i)
        {
                input.random();
                switch (m_type)
                {
                case affine_task_type::regression:
                        target.vector() = weights.matrix() * input.vector() + bias.vector();
                        add_random(rng_noise, target);
                        break;

                case affine_task_type::classification:
                        target.vector() = weights.matrix() * input.vector();
                        add_random(rng_noise, target);
                        target.vector() = class_target(target.vector());
                        break;

                default:
                        assert(false);
                }

                const auto hash = i;
                const auto label = string_t();

                add_chunk(input, hash);
                add_sample(make_fold(0), i, target, label);
        }

        return true;
}
