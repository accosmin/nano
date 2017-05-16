#include "vision/warp.h"
#include "iterator_warp.h"

namespace nano
{
        iterator_warp_t::iterator_warp_t(const string_t& config) :
                iterator_t(to_params(config,
                "type", to_string(warp_type::mixed) + "[" + concatenate(enum_values<warp_type>()) + "]",
                "noise", "0.1[0,1]",
                "sigma", "4.0[0,10]",
                "alpha", "1.0[0,10]",
                "beta", "1.0[0,10]"))
        {
        }

        tensor3d_t iterator_warp_t::input(const task_t& task, const fold_t& fold, const size_t index) const
        {
                tensor3d_t iodata = task.input(fold, index);

                const auto wtype = from_params<warp_type>(config(), "type");
                const auto noise = from_params<scalar_t>(config(), "noise");
                const auto sigma = from_params<scalar_t>(config(), "sigma");
                const auto alpha = from_params<scalar_t>(config(), "alpha");
                const auto beta = from_params<scalar_t>(config(), "beta");
                warp(iodata, wtype, noise, sigma, alpha, beta);

                return iodata;
        }

        tensor3d_t iterator_warp_t::target(const task_t& task, const fold_t& fold, const size_t index) const
        {
                return task.target(fold, index);
        }
}
