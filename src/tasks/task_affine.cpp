#include "task_affine.h"
#include "tensor/numeric.h"

using namespace nano;

affine_task_t::affine_task_t() :
        mem_tensor_task_t(make_dims(32, 1, 1), make_dims(32, 1, 1), 10)
{
}

void affine_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "isize", m_isize, "osize", m_osize, "noise", m_noise, "count", m_count, "folds", m_folds);
        reconfig(make_dims(m_isize, 1, 1), make_dims(m_osize, 1, 1), m_folds);
}

void affine_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "isize", m_isize, "osize", m_osize, "noise", m_noise, "count", m_count, "folds", m_folds);
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

                const auto hash = i;
                add_chunk(input, hash);
        }

        // generate folds
        for (size_t f = 0; f < m_folds; ++ f)
        {
                const auto protocols = split3(m_count, protocol::train, 40, protocol::valid, 30, protocol::test);

                for (size_t i = 0; i < m_count; ++ i)
                {
                        target.vector() = weights.matrix() * chunk(i).vector() + bias.vector();
                        add_random(udist_noise, rng, target);

                        const auto label = string_t();
                        const auto fold = fold_t{f, protocols[i]};

                        add_sample(fold, i, target, label);
                }
        }

        return true;
}
