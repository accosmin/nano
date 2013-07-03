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
                tanh_activation_t(const string_t& /*params*/ = string_t())
                        :       activation_t("hyperbolic tangent activation function")
                {
                }

                // create an object clone
                virtual ractivation_t clone(const string_t& params) const
                {
                        return ractivation_t(new tanh_activation_t(params));
                }

                // output & gradient
                virtual scalar_t value(scalar_t x) const { return _value(x); }
                virtual scalar_t vgrad(scalar_t x) const { return _vgrad(x); }

        private:

                // helper functions
                scalar_t _value(scalar_t x) const
                {
                        const scalar_t pexp = exp(x);
                        const scalar_t nexp = exp(-x);//1.0 / pexp;
                        return (pexp - nexp) / (pexp + nexp);
                }

                scalar_t _vgrad(scalar_t x) const
                {
                        const scalar_t vx = _value(x);
                        return vx * (1.0 - vx);
                }
        };
}

#endif // NANOCV_ACTIVATION_TANH_H
