#ifndef NANOCV_TANH_ACTIVATION_LAYER_H
#define NANOCV_TANH_ACTIVATION_LAYER_H

#include "layer_activation.h"

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

                // short description
                virtual string_t describe() const
                {
                        return (boost::format("tanh (%1%x%2%x%3%) -> %4%%5%%6%")
                                % n_idims() % n_irows() % n_icols()
                                % n_odims() % n_orows() % n_ocols()).str();
                }

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
