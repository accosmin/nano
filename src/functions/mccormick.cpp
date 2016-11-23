#include "mccormick.h"
#include <cmath>

namespace nano
{
        function_mccormick_t::function_mccormick_t() :
                function_t("McCormick", 2, 2, 2, convexity::no, 4)
        {
        }

        scalar_t function_mccormick_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto a = x(0), b = x(1);

                if (gx)
                {
                        (*gx)(0) = std::cos(a + b) + 2 * (a - b) - scalar_t(1.5);
                        (*gx)(1) = std::cos(a + b) - 2 * (a - b) + scalar_t(2.5);
                }

                return std::sin(a + b) + (a - b) * (a - b) - scalar_t(1.5) * a + scalar_t(2.5) * b + 1;
        }
}
