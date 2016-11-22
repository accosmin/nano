#include "goldstein_price.h"

namespace nano
{
        function_goldstein_price_t::function_goldstein_price_t() :
                test_function_t("Goldstein-Price", 2, 2, 2, convexity::no, 2)
        {
        }

        scalar_t function_goldstein_price_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto a = x(0), b = x(1);

                const auto z0 = 1 + a + b;
                const auto z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
                const auto z2 = 2 * a - 3 * b;
                const auto z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

                const auto u = 1 + z0 * z0 * z1;
                const auto v = 30 + z2 * z2 * z3;

                const auto z0da = 1;
                const auto z0db = 1;

                const auto z1da = -14 + 6 * a + 6 * b;
                const auto z1db = -14 + 6 * a + 6 * b;

                const auto z2da = +2;
                const auto z2db = -3;

                const auto z3da = -32 + 24 * a - 36 * b;
                const auto z3db = +48 - 36 * a + 54 * b;

                if (gx)
                {
                        (*gx)(0) = u * z2 * (2 * z2da * z3 + z2 * z3da) + v * z0 * (2 * z0da * z1 + z0 * z1da);
                        (*gx)(1) = u * z2 * (2 * z2db * z3 + z2 * z3db) + v * z0 * (2 * z0db * z1 + z0 * z1db);
                }

                return u * v;
        }
}
