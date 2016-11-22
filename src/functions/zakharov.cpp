#include "zakharov.h"
#include "math/numeric.h"

namespace nano
{
        function_zakharov_t::function_zakharov_t(const tensor_size_t dims) :
                test_function_t("Zakharov", dims, 2, 100 * 1000, convexity::yes, 5)
        {
        }

        scalar_t function_zakharov_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const vector_t bias = vector_t::LinSpaced(size(), scalar_t(0.5), scalar_t(size()) / scalar_t(2));

                const scalar_t u = x.array().square().sum();
                const scalar_t v = (bias.array() * x.array()).sum();

                if (gx)
                {
                        *gx = 2 * x.array() + (2 * v + 4 * nano::cube(v)) * bias.array();
                }

                return u + nano::square(v) + nano::quartic(v);
        }
}
