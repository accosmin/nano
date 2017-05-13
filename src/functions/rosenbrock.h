#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Rosenbrock test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_rosenbrock_t final : public function_t
        {
                explicit function_rosenbrock_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
