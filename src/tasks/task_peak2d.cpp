#include "task_peak2d.h"
#include "math/numeric.h"

using namespace nano;

peak2d_task_t::peak2d_task_t() :
        mem_tensor_task_t(make_dims(1, 32, 32), make_dims(2, 1, 1), 10)
{
}

void peak2d_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "irows", m_irows, "icols", m_icols, "noise", m_noise, "count", m_count, "folds", m_folds);
        reconfig(make_dims(1, m_irows, m_icols), make_dims(2, 1, 1), m_folds);
}

void peak2d_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "irows", m_irows, "icols", m_icols, "noise", m_noise, "count", m_count, "folds", m_folds);
}

bool peak2d_task_t::populate()
{
        auto rng = make_rng();
        auto udist_noise = make_udist<scalar_t>(-m_noise, +m_noise);
        auto udist_peakx = make_udist<tensor_size_t>(0, m_icols - 1);
        auto udist_peaky = make_udist<tensor_size_t>(0, m_irows - 1);

        tensor3d_t input(1, m_irows, m_icols);
        tensor4d_t targets(static_cast<tensor_size_t>(m_count), 2, 1, 1);

        // generate samples
        reserve_chunks(m_count);
        for (size_t i = 0; i < m_count; ++ i)
        {
                const auto peakx = udist_peakx(rng);
                const auto peaky = udist_peaky(rng);

                for (tensor_size_t y = 0; y < m_irows; ++ y)
                {
                        for (tensor_size_t x = 0; x < m_icols; ++ x)
                        {
                                const auto dx = static_cast<scalar_t>(x - peakx) / static_cast<scalar_t>(m_icols);
                                const auto dy = static_cast<scalar_t>(y - peaky) / static_cast<scalar_t>(m_irows);

                                input(0, y, x) = square(dx) + square(dy) + udist_noise(rng);
                        }
                }

                const auto ti = static_cast<tensor_size_t>(i);
                targets(ti, 0, 0, 0) = static_cast<scalar_t>(peakx) / static_cast<scalar_t>(m_icols);
                targets(ti, 1, 0, 0) = static_cast<scalar_t>(peaky) / static_cast<scalar_t>(m_irows);

                const auto hash = i;
                add_chunk(input, hash);
        }

        // generate folds
        for (size_t f = 0; f < m_folds; ++ f)
        {
                const auto protocols = split3(m_count, protocol::train, 40, protocol::valid, 30, protocol::test);

                for (size_t i = 0; i < m_count; ++ i)
                {
                        const auto label = string_t();
                        const auto fold = fold_t{f, protocols[i]};
                        const auto ti = static_cast<tensor_size_t>(i);

                        add_sample(fold, i, targets.tensor(ti), label);
                }
        }

        return true;
}
