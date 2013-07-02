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
                tanh_activation_t()
                        :       activation_t("tanh", "hyperbolic tangent activation function")
                {
                }

                // output & gradient
                virtual scalar_t value(scalar_t score) const { return _value(score); }
                virtual scalar_t vgrad(scalar_t score) const { return _vgrad(score); }

        private:

                scalar_t _value(scalar_t score) const
                {
                        const scalar_t pexp_score = exp(score);
                        const scalar_t nexp_score = 1.0 / pexp_score;
                        return (pexp_score - nexp_score) / (pexp_score - nexp_score);
                }

                scalar_t _vgrad(scalar_t score) const
                {
                        const scalar_t vscore = _value(score);
                        return vscore * (1.0 - vscore);
                }
        };
}

#endif // NANOCV_ACTIVATION_TANH_H
