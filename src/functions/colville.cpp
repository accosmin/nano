#include "colville.h"
#include "math/numeric.h"

namespace nano
{
        function_colville_t::function_colville_t() :
                test_function_t("Colville", 4, 4, 4, convexity::no, 10)
        {
        }

        scalar_t function_colville_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto x1 = x(0);
                const auto x2 = x(1);
                const auto x3 = x(2);
                const auto x4 = x(3);

                if (gx)
                {
                        (*gx)(0) = 400 * (x1 * x1 - x2) * x1 + 2 * (x1 - 1);
                        (*gx)(1) = -200 * (x1 * x1 - x2) + scalar_t(20.2) * (x2 - 1) + scalar_t(19.8) * (x4 - 1);
                        (*gx)(2) = 360 * (x3 * x3 - x4) * x3 + 2 * (x3 - 1);
                        (*gx)(3) = -180 * (x3 * x3 - x4) + scalar_t(20.2) * (x4 - 1) + scalar_t(19.8) * (x2 - 1);
                }

                return  100 * nano::square(x1 * x1 - x2) +
                        nano::square(x1 - 1) +
                        nano::square(x3 - 1) +
                        90 * nano::square(x3 * x3 - x4) +
                        scalar_t(10.1) * nano::square(x2 - 1) +
                        scalar_t(10.1) * nano::square(x4 - 1) +
                        scalar_t(19.8) * (x2 - 1) * (x4 - 1);
        }
}
