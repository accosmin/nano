#include "3hump_camel.h"

namespace nano
{
        function_3hump_camel_t::function_3hump_camel_t() :
                test_function_t("3hump camel", 2, 2, 2, convexity::no, 5)
        {
        }

        scalar_t function_3hump_camel_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto a = x(0), b = x(1);

                const auto a2 = a * a;
                const auto a3 = a * a2;
                const auto a4 = a * a3;
                const auto a5 = a * a4;
                const auto a6 = a * a5;

                if (gx)
                {
                        (*gx)(0) = scalar_t(4) * a - scalar_t(1.05) * scalar_t(4) * a3 + a5 + b;
                        (*gx)(1) = a + scalar_t(2) * b;
                }

                return scalar_t(2) * a2 - scalar_t(1.05) * a4 + a6 / scalar_t(6.0) + a * b + b * b;
        }
}
