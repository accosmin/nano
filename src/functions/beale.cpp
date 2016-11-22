#include "beale.h"

namespace nano
{
        function_beale_t::function_beale_t() :
                test_function_t("Beale", 2, 2, 2, convexity::no, scalar_t(4.5))
        {
        }

        scalar_t function_beale_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto a = x(0);
                const auto b = x(1), b2 = b * b, b3 = b2 * b;

                const auto z0 = scalar_t(1.5) - a + a * b;
                const auto z1 = scalar_t(2.25) - a + a * b2;
                const auto z2 = scalar_t(2.625) - a + a * b3;

                if (gx)
                {
                        (*gx)(0) = 2 * (z0 * (-1 + b) + z1 * (-1 + b2) + z2 * (-1 + b3));
                        (*gx)(1) = 2 * (z0 * (a) + z1 * (2 * a * b) + z2 * (3 * a * b2));
                }

                return z0 * z0 + z1 * z1 + z2 * z2;
        }
}
