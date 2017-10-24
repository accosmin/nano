#include "vision/warp.h"
#include "enhancer_warp.h"

using namespace nano;

enhancer_warp_t::enhancer_warp_t(const string_t& config) :
        enhancer_t(to_params(config,
        "type", to_string(warp_type::mixed) + "[" + concatenate(enum_values<warp_type>()) + "]",
        "noise", "0.1[0,1]",
        "sigma", "4.0[0,10]",
        "alpha", "1.0[0,10]",
        "beta", "1.0[0,10]"))
{
}

minibatch_t enhancer_warp_t::get(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
{
        const auto wtype = from_params<warp_type>(config(), "type");
        const auto noise = from_params<scalar_t>(config(), "noise");
        const auto sigma = from_params<scalar_t>(config(), "sigma");
        const auto alpha = from_params<scalar_t>(config(), "alpha");
        const auto beta = from_params<scalar_t>(config(), "beta");

        minibatch_t minibatch = task.get(fold, begin, end);
        warp(minibatch.idata(), wtype, noise, sigma, alpha, beta);

        return minibatch;
}
