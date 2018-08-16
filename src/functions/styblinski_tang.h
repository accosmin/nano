#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Styblinski-Tang function: see https://www.sfu.ca/~ssurjano/stybtang.html.
        ///
        class function_styblinski_tang_t final : public function_t
        {
        public:

                explicit function_styblinski_tang_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
