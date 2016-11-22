#include "matyas.h"

namespace nano
{
        function_matyas_t::function_matyas_t() :
                test_function_t("Matyas", 2, 2, 2, convexity::yes, 10)
        {
        }

        scalar_t function_matyas_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto a = x(0), b = x(1);

                if (gx)
                {
                        (*gx)(0) = scalar_t(0.26) * 2 * a - scalar_t(0.48) * b;
                        (*gx)(1) = scalar_t(0.26) * 2 * b - scalar_t(0.48) * a;
                }

                return scalar_t(0.26) * (a * a + b * b) - scalar_t(0.48) * a * b;
        }
}
