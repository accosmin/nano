#ifndef NANOCV_SNORM_ACTIVATION_LAYER_H
#define NANOCV_SNORM_ACTIVATION_LAYER_H

#include "layer_activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // x/sqrt(1+x^2) activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        namespace impl
        {
                struct snorm_activation_layer_eval_t
                {
                        scalar_t operator()(scalar_t x) const
                        {
                                return x / sqrt(1.0 + x * x);
                        }
                };

                struct snorm_activation_layer_grad_t
                {
                        scalar_t operator()(scalar_t g, scalar_t o) const
                        {
                                const scalar_t d = 1.0 - o * o;
                                return g * d * sqrt(d);
                        }
                };
        }

        class snorm_activation_layer_t : public activation_layer_t
        <
                impl::snorm_activation_layer_eval_t,
                impl::snorm_activation_layer_grad_t
        >
        {
        public:

                // constructor
                snorm_activation_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(snorm_activation_layer_t, layer_t, "x/sqrt(1+x^2) activation layer")
        };
}

#endif // NANOCV_SNORM_ACTIVATION_LAYER_H
