#include "vision/warp.h"
#include "enhancer_warp.h"

using namespace nano;

namespace
{
        template <typename titensor, typename trng>
        void apply(titensor&& idata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta, trng&& rng)
        {
                warp(idata, wtype, noise, sigma, alpha, beta, rng);
        }
}

enhancer_warp_t::enhancer_warp_t(const string_t& config) :
        enhancer_t(to_params(config,
        "type", to_string(warp_type::mixed) + "[" + concatenate(enum_values<warp_type>()) + "]",
        "noise", "0.1[0,1]",
        "sigma", "4.0[0,10]",
        "alpha", "1.0[0,10]",
        "beta", "1.0[0,10]"))
{
}

sample_t enhancer_warp_t::get(const task_t& task, const fold_t& fold, const size_t index) const
{
        const auto wtype = from_params<warp_type>(config(), "type");
        const auto noise = from_params<scalar_t>(config(), "noise");
        const auto sigma = from_params<scalar_t>(config(), "sigma");
        const auto alpha = from_params<scalar_t>(config(), "alpha");
        const auto beta = from_params<scalar_t>(config(), "beta");

        auto rng = make_rng<scalar_t>(-1, +1);

        sample_t sample = task.get(fold, index);
        apply(sample.m_input, wtype, noise, sigma, alpha, beta, rng);

        return sample;
}

minibatch_t enhancer_warp_t::get(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
{
        const auto wtype = from_params<warp_type>(config(), "type");
        const auto noise = from_params<scalar_t>(config(), "noise");
        const auto sigma = from_params<scalar_t>(config(), "sigma");
        const auto alpha = from_params<scalar_t>(config(), "alpha");
        const auto beta = from_params<scalar_t>(config(), "beta");

        auto rng = make_rng<scalar_t>(-1, +1);

        minibatch_t minibatch = task.get(fold, begin, end);
        for (auto index = 0; index < minibatch.count(); ++ index)
        {
                apply(minibatch.idata(index), wtype, noise, sigma, alpha, beta, rng);
        }

        return minibatch;
}
