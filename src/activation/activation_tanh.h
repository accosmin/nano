#ifndef NANOCV_ACTIVATION_TANH_H
#define NANOCV_ACTIVATION_TANH_H

#include "activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // hyperbolic tangent activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class tanh_activation_t : public activation_t
        {
        public:

                // constructor
                tanh_activation_t(const string_t& /*params*/ = string_t()) {}

                NCV_MAKE_CLONABLE(tanh_activation_t, activation_t, "hyperbolic tangent activation function")

                // output & gradient (as a function of the output)
                virtual scalar_t value(scalar_t x) const { return _value(x); }
                virtual scalar_t vgrad(scalar_t x) const { return _vgrad(x); }

        private:

                // helper functions
                scalar_t _value(scalar_t x) const
                {
                        const scalar_t pexp = exp(x), nexp = 1.0 / pexp;
                        return (pexp - nexp) / (pexp + nexp);
                }

                scalar_t _vgrad(scalar_t x) const
                {
                        return 1.0 - x * x;
                }
        };
}

#endif // NANOCV_ACTIVATION_TANH_H
