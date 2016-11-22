#pragma once

#include "test_function.h"

namespace nano
{
        ///
        /// \brief create Schumer-Steiglitz No. 02 test functions
        ///
        struct function_schumer_steiglitz_t final : public test_function_t
        {
                explicit function_schumer_steiglitz_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
