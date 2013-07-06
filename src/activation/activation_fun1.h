#ifndef NANOCV_ACTIVATION_FUN1_H
#define NANOCV_ACTIVATION_FUN1_H

#include "activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // x/sqrt(1+x^2) activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class fun1_activation_t : public activation_t
        {
        public:

                // constructor
                fun1_activation_t(const string_t& /*params*/ = string_t())
                        :       activation_t("x/sqrt(1+x^2) activation function")
                {
                }

                // create an object clone
                virtual ractivation_t clone(const string_t& params) const
                {
                        return ractivation_t(new fun1_activation_t(params));
                }

                // output & gradient (as a function of the output)
                virtual scalar_t value(scalar_t x) const { return _value(x); }
                virtual scalar_t vgrad(scalar_t x) const { return _vgrad(x); }

        private:

                // helper functions
                scalar_t _value(scalar_t x) const
                {
                        return x / sqrt(1.0 + x * x);
                }

                scalar_t _vgrad(scalar_t x) const
                {
                        const scalar_t d = 1.0 - x * x;
                        return d * sqrt(d);
                }
        };
}

#endif // NANOCV_ACTIVATION_FUN1_H
