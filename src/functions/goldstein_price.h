#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Goldstein-Price test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_goldstein_price_t final : public function_t
        {
                function_goldstein_price_t();

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
