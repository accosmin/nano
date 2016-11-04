#include "task_affine.h"
#include "math/random.h"
#include "math/numeric.h"
#include "tensor/numeric.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        static string_t append_config(const string_t& configuration)
        {
                return  concat_params(configuration,
                        "idims=10[1,100],irows=32[1,100],icols=32[1,100],osize=10[1,1000],"\
                        "count=1000[10,1M],noise=0.1[0.001,0.5]");
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
                const auto noise = clamp(from_params<scalar_t>(config(), "noise", scalar_t(0.1)), scalar_t(0.001), scalar_t(0.5));

                random_t<scalar_t> rng_input(-scalar_t(1.0), +scalar_t(1.0));
                random_t<scalar_t> rng_noise(-noise, +noise);

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

                        // affine target
                        vector_t target = m_A * input.vector() + m_b;
                        tensor::add_random(rng_noise, target);

                        add_sample(make_fold(0), i, target);
                }

                return true;
        }
}
