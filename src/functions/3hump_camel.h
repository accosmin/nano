#pragma once

#include "test_function.h"

namespace nano
{
        ///
        /// \brief create three-hump camel test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_3hump_camel_t final : public test_function_t
        {
                function_3hump_camel_t();

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
