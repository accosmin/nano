#ifndef NANOCV_TANH_ACTIVATION_LAYER_H
#define NANOCV_TANH_ACTIVATION_LAYER_H

#include "layer_activation.h"

namespace ncv
{        
        namespace impl
        {
                struct tanh_activation_layer_eval_t
                {
                        scalar_t operator()(scalar_t x) const
                        {
                                const scalar_t pexp = exp(x), nexp = 1.0 / pexp;
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
        /// hyperbolic tangent activation function
        ///
        class tanh_activation_layer_t : public activation_layer_t
        <
                impl::tanh_activation_layer_eval_t,
                impl::tanh_activation_layer_grad_t
        >
        {
        public:

                // constructor
                tanh_activation_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(tanh_activation_layer_t, layer_t, "hyperbolic tangent activation layer")
        };
}

#endif // NANOCV_TANH_ACTIVATION_LAYER_H
