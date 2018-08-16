#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Rosenbrock function: see https://en.wikipedia.org/wiki/Test_functions_for_optimization.
        ///
        class function_rosenbrock_t final : public function_t
        {
        public:

                explicit function_rosenbrock_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
