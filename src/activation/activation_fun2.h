#ifndef NANOCV_ACTIVATION_FUN2_H
#define NANOCV_ACTIVATION_FUN2_H

#include "activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // x/(1+|x|) activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class fun2_activation_t : public activation_t
        {
        public:

                // constructor
                fun2_activation_t(const string_t& /*params*/ = string_t())
                        :       activation_t("x/(1+|x|) activation function")
                {
                }

                // create an object clone
                virtual ractivation_t clone(const string_t& params) const
                {
                        return ractivation_t(new fun2_activation_t(params));
                }

                // output & gradient (as a function of the output)
                virtual scalar_t value(scalar_t x) const { return _value(x); }
                virtual scalar_t vgrad(scalar_t x) const { return _vgrad(x); }

        private:

                // helper functions
                scalar_t _value(scalar_t x) const
                {
                        return x / (1.0 + std::fabs(x));
                }

                scalar_t _vgrad(scalar_t x) const
                {
                        const scalar_t d = (x > 0.0) ? (1.0 - x) : (1.0 + x);
                        return d * d;
                }
        };
}

#endif // NANOCV_ACTIVATION_FUN2_H
