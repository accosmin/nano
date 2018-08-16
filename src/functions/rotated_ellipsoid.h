#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief rotated hyper-ellipsoid function: see https://www.sfu.ca/~ssurjano/rothyp.html.
        ///
        class function_rotated_ellipsoid_t final : public function_t
        {
        public:

                explicit function_rotated_ellipsoid_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
