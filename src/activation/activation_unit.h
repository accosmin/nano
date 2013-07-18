#ifndef NANOCV_ACTIVATION_UNIT_H
#define NANOCV_ACTIVATION_UNIT_H

#include "activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // identity activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
        class unit_activation_t : public activation_t
        {
        public:

                // constructor
                unit_activation_t(const string_t& /*params*/ = string_t()) {}

                NCV_MAKE_CLONABLE(unit_activation_t, activation_t, "identity activation function")

                // output & gradient (as a function of the output)
                virtual scalar_t value(scalar_t x) const { return x; }
                virtual scalar_t vgrad(scalar_t x) const { return 1.0; }
        };
}

#endif // NANOCV_ACTIVATION_UNIT_H
