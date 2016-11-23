#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Himmelblau test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_himmelblau_t final : public function_t
        {
                function_himmelblau_t();

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
