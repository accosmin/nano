#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief sphere function: f(x) = x.dot(x).
        ///
        class function_sphere_t final : public function_t
        {
        public:

                explicit function_sphere_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
