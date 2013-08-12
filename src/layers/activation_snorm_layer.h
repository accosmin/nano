#ifndef NANOCV_SNORM_ACTIVATION_LAYER_H
#define NANOCV_SNORM_ACTIVATION_LAYER_H

#include "activation_layer.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // x/sqrt(1+x^2) activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class snorm_activation_layer_t : public activation_layer_t
        {
        public:

                // constructor
                snorm_activation_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(snorm_activation_layer_t, layer_t, "x/sqrt(1+x^2) activation layer")

        protected:

                // activation outputs & gradients
                virtual scalar_t forward_scalar(scalar_t ix) const
                {
                        return ix / sqrt(1.0 + ix * ix);
                }
                virtual scalar_t backward_scalar(scalar_t gx, scalar_t ox) const
                {
                        const scalar_t d = 1.0 - ox * ox;
                        return gx * d * sqrt(d);
                }
        };
}

#endif // NANOCV_SNORM_ACTIVATION_LAYER_H
