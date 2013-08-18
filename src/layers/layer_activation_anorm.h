#ifndef NANOCV_ANORM_ACTIVATION_LAYER_H
#define NANOCV_ANORM_ACTIVATION_LAYER_H

#include "layer_activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // x/(1+|x|) activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class anorm_activation_layer_t : public activation_layer_t
        {
        public:

                // constructor
                anorm_activation_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(anorm_activation_layer_t, layer_t, "x/(1+|x|) activation layer")

        protected:

                // activation outputs & gradients
                virtual scalar_t value(scalar_t ix) const
                {
                        return ix / (1.0 + std::fabs(ix));
                }
                virtual scalar_t vgrad(scalar_t gx, scalar_t ox) const
                {
                        const scalar_t d = (ox > 0.0) ? (1.0 - ox) : (1.0 + ox);
                        return gx * d * d;
                }
        };
}

#endif // NANOCV_ANORM_ACTIVATION_LAYER_H
