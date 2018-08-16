#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D).
        ///
        class function_axis_ellipsoid_t final : public function_t
        {
        public:

                explicit function_axis_ellipsoid_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
