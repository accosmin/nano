#include "task_affine.h"
#include "tensor/numeric.h"

using namespace nano;

affine_task_t::affine_task_t() :
        mem_tensor_task_t(make_dims(32, 1, 1), make_dims(32, 1, 1), 1)
{
}

void affine_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "isize", m_isize, "osize", m_osize, "noise", m_noise, "count", m_count);
        reconfig(make_dims(m_isize, 1, 1), make_dims(m_osize, 1, 1), 1);
}

void affine_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "isize", m_isize, "osize", m_osize, "noise", m_noise, "count", m_count);
}

bool affine_task_t::populate()
{
        auto rng = make_rng();
        auto udist_noise = make_udist<scalar_t>(-m_noise, +m_noise);

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
                target.vector() = weights.matrix() * input.vector() + bias.vector();
                add_random(udist_noise, rng, target);

                const auto hash = i;
                const auto label = string_t();

                add_chunk(input, hash);
                add_sample(make_fold(0), i, target, label);
        }

        return true;
}
