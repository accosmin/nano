#ifndef NANOCV_ACTIVATION_H
#define NANOCV_ACTIVATION_H

#include "core/manager.h"

namespace ncv
{
        // manage activation functions (register new ones, query and clone them)
        class activation_t;
        typedef manager_t<activation_t>         activation_manager_t;
        typedef activation_manager_t::robject_t ractivation_t;

        ////////////////////////////////////////////////////////////////////////////////
        // generic activation function used for transforming a scalar
        //      (e.g. some model's output).
        ////////////////////////////////////////////////////////////////////////////////
	
        class activation_t : public clonable_t<activation_t>
        {
        public:

                // constructor
                activation_t(const string_t& description)
                        :       clonable_t<activation_t>(description)
                {
                }

                // destructor
                virtual ~activation_t() {}

                // output & gradient
                virtual scalar_t value(scalar_t x) const = 0;
                virtual scalar_t vgrad(scalar_t x) const = 0;
        };
}

#endif // NANOCV_ACTIVATION_H
