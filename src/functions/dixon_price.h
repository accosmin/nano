#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Dixon-Price function: see https://www.sfu.ca/~ssurjano/dixonpr.html.
        ///
        class function_dixon_price_t final : public function_t
        {
        public:

                explicit function_dixon_price_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
