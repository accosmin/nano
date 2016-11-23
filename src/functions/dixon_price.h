#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Dixon-Price test functions
        ///
        struct function_dixon_price_t final : public function_t
        {
                explicit function_dixon_price_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
