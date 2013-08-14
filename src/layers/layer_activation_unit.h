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

                // short description
                virtual string_t describe() const
                {
                        return (boost::format("unit (%1%x%2%x%3%) -> %4%x%5%x%6%")
                                % n_idims() % n_irows() % n_icols()
                                % n_odims() % n_orows() % n_ocols()).str();
                }

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
