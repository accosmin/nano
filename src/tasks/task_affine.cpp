#include "class.h"
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
                        "idims", "10[1,100]", "irows", "32[1,100]", "icols", "32[1,100]", "osize", "10[1,1000]",
                        "count", "1000[10,1M]", "noise", "0.1[0,0.5]", "mode", "regression[,sign_class]");
        }

        affine_task_t::affine_task_t(const string_t& configuration) : mem_tensor_task_t(
                "affine",
                clamp(from_params<tensor_size_t>(append_config(configuration), "idims", 10), 1, 100),
                clamp(from_params<tensor_size_t>(append_config(configuration), "irows", 32), 1, 100),
                clamp(from_params<tensor_size_t>(append_config(configuration), "icols", 32), 1, 100),
                clamp(from_params<tensor_size_t>(append_config(configuration), "osize", 10), 1, 1000),
                1, append_config(configuration))
        {
        }

        bool affine_task_t::populate()
        {
                const auto count = clamp(from_params<size_t>(config(), "count", 1000), 10, 100000);
                const auto noise = clamp(from_params<scalar_t>(config(), "noise", scalar_t(0.1)), epsilon0<scalar_t>(), scalar_t(0.5));
                const auto mode = from_params<affine_mode>(config(), "mode");

                auto rng_input = make_rng<scalar_t>(-scalar_t(1.0), +scalar_t(1.0));
                auto rng_noise = make_rng<scalar_t>(-noise, +noise);

                // random affine transformation
                const auto isize = idims() * irows() * icols();

                m_A.resize(osize(), isize);
                m_b.resize(osize());

                tensor::set_random(rng_input, m_A, m_b);
                m_A /= static_cast<scalar_t>(isize);

                // generate samples
                for (size_t i = 0; i < count; ++ i)
                {
                        // random input
                        tensor3d_t input(idims(), irows(), icols());
                        tensor::set_random(rng_input, input);

                        add_chunk(input, i);

                        // target
                        vector_t target = m_A * input.vector() + m_b;
                        tensor::add_random(rng_noise, target);
                        switch (mode)
                        {
                        case affine_mode::regression:
                                add_sample(make_fold(0), i, target);
                                break;

                        case affine_mode::sign_class:
                        default:
                                add_sample(make_fold(0), i, class_target(target));
                                break;
                        }
                }

                return true;
        }
}
