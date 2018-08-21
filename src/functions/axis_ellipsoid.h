#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D).
        ///
        class function_axis_ellipsoid_t final : public function_t
        {
        public:

                explicit function_axis_ellipsoid_t(const tensor_size_t dims) :
                        function_t("Axis Parallel Hyper-Ellipsoid", dims, 1, 100 * 1000, convexity::yes, 100)
                {
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        const vector_t bias = vector_t::LinSpaced(size(), scalar_t(1), scalar_t(size()));

                        if (gx)
                        {
                                *gx = 2 * x.array() * bias.array();
                        }

                        return (x.array().square() * bias.array()).sum();
                }
        };
}
