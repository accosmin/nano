#include "sphere.h"

namespace nano
{
        function_sphere_t::function_sphere_t(const tensor_size_t dims) :
                test_function_t("Sphere", dims, 1, 100 * 1000, convexity::yes, 5)
        {
        }

        scalar_t function_sphere_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                if (gx)
                {
                        *gx = 2 * x;
                }

                return x.array().square().sum();
        }
}
