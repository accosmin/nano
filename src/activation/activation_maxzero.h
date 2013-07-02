#ifndef NANOCV_ACTIVATION_MAXZERO_H
#define NANOCV_ACTIVATION_MAXZERO_H

#include "activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // trim-to-zero activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class maxzero_activation_t : public activation_t
        {
        public:

                // constructor
                maxzero_activation_t()
                        :       activation_t("maxzero", "max(0, x) activation function")
                {
                }

                // output & gradient
                virtual scalar_t value(scalar_t score) const { return std::max(0.0, score); }
                virtual scalar_t vgrad(scalar_t score) const { return score > 0.0 ? 1.0 : 0.0; }
        };
}

#endif // NANOCV_ACTIVATION_MAXZERO_H
