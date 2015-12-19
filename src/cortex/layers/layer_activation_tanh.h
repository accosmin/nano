#pragma once

#include "layer_activation.h"

namespace cortex
{        
        namespace detail
        {
                struct tanh_activation_layer_eval_t
                {
                        scalar_t operator()(scalar_t x) const
                        {
                                const scalar_t pexp = std::exp(x), nexp = 1.0 / pexp;
                                return (pexp - nexp) / (pexp + nexp);
                        }
                };

                struct tanh_activation_layer_grad_t
                {
                        scalar_t operator()(scalar_t g, scalar_t o) const
                        {
                                return g * (1.0 - o * o);
                        }
                };
        }

        ///
        /// \brief hyperbolic tangent activation function
        ///
        class tanh_activation_layer_t : public activation_layer_t
        <
                detail::tanh_activation_layer_eval_t,
                detail::tanh_activation_layer_grad_t
        >
        {
        public:

                NANOCV_MAKE_CLONABLE(tanh_activation_layer_t, "hyperbolic tangent activation layer")

                // constructor
                explicit tanh_activation_layer_t(const string_t& parameters = string_t())
                        :       activation_layer_t(parameters)
                {
                }
        };
}
