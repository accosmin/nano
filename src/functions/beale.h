#pragma once

#include "test_function.h"

namespace nano
{
        ///
        /// \brief create Beale test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_beale_t final : public test_function_t
        {
                function_beale_t();

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
