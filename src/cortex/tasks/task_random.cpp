#include "task_random.h"
#include "cortex/class.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "tensor/random.hpp"
#include "text/to_string.hpp"
#include "text/from_params.hpp"

namespace nano
{
        random_task_t::random_task_t(const string_t& configuration) : mem_tensor_task_t(
                "random",
                nano::clamp(nano::from_params<tensor_size_t>(configuration, "idims", 10), 1, 100),
                nano::clamp(nano::from_params<tensor_size_t>(configuration, "irows", 32), 1, 100),
                nano::clamp(nano::from_params<tensor_size_t>(configuration, "icols", 32), 1, 100),
                nano::clamp(nano::from_params<tensor_size_t>(configuration, "osize", 10), 1, 100),
                nano::clamp(nano::from_params<size_t>(configuration, "folds", 1), 1, 10)),
                m_count(nano::clamp(nano::from_params<size_t>(configuration, "count", 1000), 10, 100000))
        {
        }

        bool random_task_t::populate(const string_t&)
        {
                nano::random_t<size_t> rng_fold(0, n_folds() - 1);
                nano::random_t<tensor_size_t> rng_osize(0, osize() - 1);
                nano::random_t<scalar_t> rng_input(-1.0, +1.0);

                // generate samples
                for (size_t i = 0; i < m_count; ++ i)
                {
                        // random input
                        tensor3d_t input(idims(), irows(), icols());
                        tensor::set_random(input, rng_input);

                        add_chunk(input);

                        // random target
                        const auto fold = make_random_fold(rng_fold());

                        const auto o = rng_osize();
                        const auto target = class_target(o, osize());
                        const auto label = string_t("class") + to_string(o);

                        add_sample(fold, i, target, label);
                }

                return true;
        }
}
