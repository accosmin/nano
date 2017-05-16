#include "math/random.h"
#include "iterator_noise.h"
#include "tensor/numeric.h"

namespace nano
{
        iterator_noise_t::iterator_noise_t(const string_t& config) :
                iterator_t(to_params(config, "noise", "0.1[0,1]"))
        {
        }

        tensor3d_t iterator_noise_t::input(const task_t& task, const fold_t& fold, const size_t index) const
        {
                tensor3d_t iodata = task.input(fold, index);

                const auto noise = from_params<scalar_t>(config(), "noise");
                add_random(make_rng<scalar_t>(-noise, +noise), iodata);

                return iodata;
        }

        tensor3d_t iterator_noise_t::target(const task_t& task, const fold_t& fold, const size_t index) const
        {
                return task.target(fold, index);
        }
}
