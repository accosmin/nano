#include "task_affine.h"
#include "math/random.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "tensor/numeric.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        static string_t append_config(const string_t& configuration)
        {
                return  to_params(configuration,
                        "isize", "100[1,1000]", "osize", "10[1,1000]", "count", "1000[10,1M]", "noise", "0.01[0,0.5]");
        }

        affine_task_t::affine_task_t(const string_t& configuration) : mem_tensor_task_t(
                dim3d_t{clamp(from_params<tensor_size_t>(append_config(configuration), "isize"), 1, 1000), 1, 1},
                dim3d_t{clamp(from_params<tensor_size_t>(append_config(configuration), "osize"), 1, 1000), 1, 1},
                1, append_config(configuration))
        {
        }

        bool affine_task_t::populate()
        {
                const auto count = clamp(from_params<size_t>(config(), "count"), 10, 1000000);
                const auto noise = clamp(from_params<scalar_t>(config(), "noise"), epsilon0<scalar_t>(), scalar_t(0.5));

                auto rng_input = make_rng<scalar_t>(-scalar_t(1.0), +scalar_t(1.0));
                auto rng_noise = make_rng<scalar_t>(-noise, +noise);

                // random affine transformation
                m_A.resize(nano::size(odims()), nano::size(idims()));
                m_b.resize(nano::size(odims()));

                nano::set_random(rng_input, m_A, m_b);
                nano::normalize(m_A);

                // generate samples
                for (size_t i = 0; i < count; ++ i)
                {
                        // random input
                        tensor3d_t input(idims());
                        nano::set_random(rng_input, input);
                        add_chunk(input, i);

                        // target
                        tensor3d_t target(odims());
                        nano::map_vector(target.data(), target.size()) = m_A * input.vector() + m_b;
                        nano::add_random(rng_noise, target);
                        add_sample(make_fold(0), i, target);
                }

                return true;
        }
}
