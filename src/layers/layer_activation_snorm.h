#ifndef NANOCV_SNORM_ACTIVATION_LAYER_H
#define NANOCV_SNORM_ACTIVATION_LAYER_H

#include "layer_activation.h"

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

                // short description
                virtual string_t describe() const
                {
                        return (boost::format("x/sqrt(1+x^2) (%1%x%2%x%3%) -> %4%x%5%x%6%")
                                % n_idims() % n_irows() % n_icols()
                                % n_odims() % n_orows() % n_ocols()).str();
                }

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
