#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Powell function: see https://www.sfu.ca/~ssurjano/powell.html.
        ///
        class function_powell_t final : public function_t
        {
        public:

                explicit function_powell_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
