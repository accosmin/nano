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

                explicit function_schumer_steiglitz_t(const tensor_size_t dims) :
                        function_t("Schumer-Steiglitz", dims, 1, 100 * 1000, convexity::yes)
                {
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        if (gx)
                        {
                                *gx = 4 * x.array().cube();
                        }

                        return x.array().square().square().sum();
                }
        };
}
