#ifndef NANOCV_ACTIVATION_XNORM_H
#define NANOCV_ACTIVATION_XNORM_H

#include "activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // x / (1 + |x|) activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class xnorm_activation_t : public activation_t
        {
        public:

                // constructor
                xnorm_activation_t(const string_t& /*params*/ = string_t())
                        :       activation_t("x / (1 + |x|) activation function")
                {
                }

                // create an object clone
                virtual ractivation_t clone(const string_t& params) const
                {
                        return ractivation_t(new xnorm_activation_t(params));
                }

                // output & gradient
                virtual scalar_t value(scalar_t x) const { return x / (1.0 + std::fabs(x)); }
                virtual scalar_t vgrad(scalar_t x) const { return _vgrad(x); }

        private:

                // helper functions
                scalar_t _vgrad(scalar_t x) const
                {
                        if (x > 0.0)
                        {
                                const scalar_t dx = 1.0 + x;
                                return 1.0 / (dx * dx);
                        }

                        else
                        {
                                const scalar_t dx = 1.0 - x;
                                return 1.0 / (dx * dx);
                        }
                }
        };
}

#endif // NANOCV_ACTIVATION_XNORM_H
