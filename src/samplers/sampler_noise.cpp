#include "math/random.h"
#include "sampler_noise.h"
#include "tensor/numeric.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        sampler_noise_t::sampler_noise_t(const string_t& config) :
                sampler_t(to_params(config, "noise", "0.1[0,1]"))
        {
        }

        tensor3d_t sampler_noise_t::input(const task_t& task, const fold_t& fold, const size_t index)
        {
                const auto noise = from_params<scalar_t>(config(), "noise");
                auto rng = make_rng<scalar_t>(-noise, +noise);

                tensor3d_t iodata = task.input(fold, index);
                add_random(rng, iodata);
                return iodata;
        }

        tensor3d_t sampler_noise_t::target(const task_t& task, const fold_t& fold, const size_t index)
        {
                return task.target(fold, index);
        }
}
