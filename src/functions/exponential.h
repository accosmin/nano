#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief exponential function: f(x) = exp(1 + x.dot(x) / D).
        ///
        class function_exponential_t final : public function_t
        {
        public:

                explicit function_exponential_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
