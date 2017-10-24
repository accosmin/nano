#include "math/random.h"
#include "enhancer_noise.h"
#include "tensor/numeric.h"

using namespace nano;

enhancer_noise_t::enhancer_noise_t(const string_t& config) :
        enhancer_t(to_params(config, "noise", "0.1[0,1]"))
{
}

minibatch_t enhancer_noise_t::get(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
{
        const auto noise = from_params<scalar_t>(config(), "noise");

        auto rng = make_rng<scalar_t>(-1, +1);

        minibatch_t minibatch = task.get(fold, begin, end);
        for (auto index = 0; index < minibatch.count(); ++ index)
        {
                auto&& idata = minibatch.idata(index);

                // keep input and add some salt & pepper noise
                nano::add_random([&] () { return rng() * noise; }, idata);
        }

        return minibatch;
}
