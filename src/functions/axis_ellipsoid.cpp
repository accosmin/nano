#include "axis_ellipsoid.h"

namespace nano
{
        function_axis_ellipsoid_t::function_axis_ellipsoid_t(const tensor_size_t dims) :
                test_function_t("Axis Parallel Hyper-Ellipsoid", dims, 1, 100 * 1000, convexity::yes, 100)
        {
        }

        scalar_t function_axis_ellipsoid_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const vector_t bias = vector_t::LinSpaced(size(), scalar_t(1), scalar_t(size()));

                if (gx)
                {
                        *gx = 2 * x.array() * bias.array();
                }

                return (x.array().square() * bias.array()).sum();
        }
}
