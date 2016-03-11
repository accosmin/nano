#pragma once

#include "layer_activation.h"

namespace nano
{
        namespace detail
        {
                struct snorm_activation_layer_eval_t
                {
                        scalar_t operator()(scalar_t x) const
                        {
                                return x / std::sqrt(1.0 + x * x);
                        }
                };

                struct snorm_activation_layer_grad_t
                {
                        scalar_t operator()(scalar_t g, scalar_t o) const
                        {
                                const scalar_t d = 1.0 - o * o;
                                return g * d * std::sqrt(d);
                        }
                };
        }

        ///
        /// \brief x/sqrt(1+x^2) activation function
        ///
        class snorm_activation_layer_t : public activation_layer_t
        <
                detail::snorm_activation_layer_eval_t,
                detail::snorm_activation_layer_grad_t
        >
        {
        public:

                NANO_MAKE_CLONABLE(snorm_activation_layer_t, "x/sqrt(1+x^2) activation layer")

                // constructor
                explicit snorm_activation_layer_t(const string_t& parameters = string_t())
                        :       activation_layer_t(parameters)
                {
                }
        };
}
