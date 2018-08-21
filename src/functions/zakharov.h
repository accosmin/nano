#pragma once

#include "function.h"
#include "core/numeric.h"

namespace nano
{
        ///
        /// \brief Zakharov function: see https://www.sfu.ca/~ssurjano/zakharov.html.
        ///
        class function_zakharov_t final : public function_t
        {
        public:

                explicit function_zakharov_t(const tensor_size_t dims) :
                        function_t("Zakharov", dims, 2, 100 * 1000, convexity::yes, 5)
                {
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
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
        };
}
