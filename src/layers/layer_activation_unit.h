#ifndef NANOCV_UNIT_ACTIVATION_LAYER_H
#define NANOCV_UNIT_ACTIVATION_LAYER_H

#include "layer_activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // identity activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class unit_activation_layer_t : public activation_layer_t
        {
        public:

                // constructor
                unit_activation_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(unit_activation_layer_t, layer_t, "identity activation layer")

        protected:

                // activation outputs & gradients
                virtual scalar_t value(scalar_t ix) const
                {
                        return ix;
                }
                virtual scalar_t vgrad(scalar_t gx, scalar_t ox) const
                {
                        return gx;
                }
        };
}

#endif // NANOCV_UNIT_ACTIVATION_LAYER_H
