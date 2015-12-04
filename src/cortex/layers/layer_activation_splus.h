#pragma once

#include "layer_activation.h"

namespace cortex
{
        namespace detail
        {
                struct softplus_activation_layer_eval_t
                {
                        scalar_t operator()(scalar_t x) const
                        {
                                return std::log1p(std::exp(x));
                        }
                };

                struct softplus_activation_layer_grad_t
                {
                        scalar_t operator()(scalar_t g, scalar_t o) const
                        {
                                return g * (1.0 - std::exp(-o));
                        }
                };
        }

        ///
        /// \brief soft-plus (max approximation) activation function
        ///
        class softplus_activation_layer_t : public activation_layer_t
        <
                detail::softplus_activation_layer_eval_t,
                detail::softplus_activation_layer_grad_t
        >
        {
        public:

                NANOCV_MAKE_CLONABLE(softplus_activation_layer_t, "soft-plus activation layer")

                // constructor
                softplus_activation_layer_t(const string_t& parameters = string_t())
                        :       activation_layer_t(parameters)
                {
                }
        };
}
