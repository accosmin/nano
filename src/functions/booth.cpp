#include "booth.h"

namespace nano
{
        function_booth_t::function_booth_t() :
                test_function_t("Booth", 2, 2, 2, convexity::no, 10)
        {
        }

        scalar_t function_booth_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto a = x(0), b = x(1);

                const auto u = a + 2 * b - 7;
                const auto v = 2 * a + b - 5;

                if (gx)
                {
                        (*gx)(0) = 2 * u + 4 * v;
                        (*gx)(1) = 4 * u + 2 * v;
                }

                return u * u + v * v;
        }
}

