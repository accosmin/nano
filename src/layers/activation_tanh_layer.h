#ifndef NANOCV_TANH_ACTIVATION_LAYER_H
#define NANOCV_TANH_ACTIVATION_LAYER_H

#include "activation_layer.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // hyperbolic tangent activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class tanh_activation_layer_t : public activation_layer_t
        {
        public:

                // constructor
                tanh_activation_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(tanh_activation_layer_t, layer_t, "hyperbolic tangent activation layer")

        protected:

                // activation outputs & gradients
                virtual scalar_t forward_scalar(scalar_t ix) const
                {
                        const scalar_t pexp = exp(ix), nexp = 1.0 / pexp;
                        return (pexp - nexp) / (pexp + nexp);
                }
                virtual scalar_t backward_scalar(scalar_t gx, scalar_t ox) const
                {
                        return gx * (1.0 - ox * ox);
                }
        };
}

#endif // NANOCV_TANH_ACTIVATION_LAYER_H
