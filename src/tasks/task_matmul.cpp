#include "class.h"
#include "task_matmul.h"
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
                        "irows", "32[1,100]", "icols", "32[1,100]", "count", "1000[10,1M]", "noise", "0.01[0,0.5]");
        }

        static tensor_size_t get_irows(const string_t& configuration)
        {
                return clamp(from_params<tensor_size_t>(append_config(configuration), "irows"), 1, 100);
        }

        static tensor_size_t get_icols(const string_t& configuration)
        {
                return clamp(from_params<tensor_size_t>(append_config(configuration), "icols"), 1, 100);
        }

        matmul_task_t::matmul_task_t(const string_t& configuration) : mem_tensor_task_t(
                dim3d_t{2, get_irows(configuration), get_icols(configuration)},
                dim3d_t{1, get_irows(configuration), get_irows(configuration)},
                1, append_config(configuration))
        {
        }

        bool matmul_task_t::populate()
        {
                const auto count = clamp(from_params<size_t>(config(), "count"), 10, 1000000);
                const auto noise = clamp(from_params<scalar_t>(config(), "noise"), epsilon0<scalar_t>(), scalar_t(0.5));

                auto rng_input = make_rng<scalar_t>(-scalar_t(1.0), +scalar_t(1.0));
                auto rng_noise = make_rng<scalar_t>(-noise, +noise);

                const auto irows = idims().size<1>();

                // random affine transformation
                m_A.resize(irows, irows);
                m_B.resize(irows, irows);

                tensor::set_random(rng_input, m_A, m_B);
                tensor::normalize(m_A);

                // generate samples
                for (size_t i = 0; i < count; ++ i)
                {
                        // random input
                        tensor3d_t input(idims());
                        tensor::set_random(rng_input, input);
                        add_chunk(input, i);

                        // target
                        matrix_t target = m_A * (input.matrix(0) * input.matrix(1).transpose()) + m_B;
                        tensor::add_random(rng_noise, target);
                        add_sample(make_fold(0), i, tensor::map_tensor(target.data(), odims()));
                }

                return true;
        }
}
