#pragma once

#include "test_function.h"

namespace nano
{
        ///
        /// \brief create Cauchy test functions
        ///
        struct function_cauchy_t final : public test_function_t
        {
                explicit function_cauchy_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
