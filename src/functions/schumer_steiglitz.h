#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Schumer-Steiglitz No. 02 function: f(x) = sum(x_i^4, i=1,D)
        ///
        class function_schumer_steiglitz_t final : public function_t
        {
        public:

                explicit function_schumer_steiglitz_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
