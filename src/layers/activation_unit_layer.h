#ifndef NANOCV_UNIT_ACTIVATION_LAYER_H
#define NANOCV_UNIT_ACTIVATION_LAYER_H

#include "activation_layer.h"

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
                virtual scalar_t forward_scalar(scalar_t ix) const
                {
                        return ix;
                }
                virtual scalar_t backward_scalar(scalar_t gx, scalar_t ox) const
                {
                        return gx;
                }
        };
}

#endif // NANOCV_UNIT_ACTIVATION_LAYER_H
