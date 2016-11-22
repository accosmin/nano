#include "himmelblau.h"

namespace nano
{
        function_himmelblau_t::function_himmelblau_t() :
                test_function_t("Himmelblau", 2, 2, 2, convexity::no, 1e+6)
        {
        }

        scalar_t function_himmelblau_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto a = x(0), b = x(1);

                const auto u = a * a + b - 11;
                const auto v = a + b * b - 7;

                if (gx)
                {
                        (*gx)(0) = 2 * u * 2 * a + 2 * v;
                        (*gx)(1) = 2 * u + 2 * v * 2 * b;
                }

                return u * u + v * v;
        }
}
