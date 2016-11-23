#include "sargan.h"
#include "math/numeric.h"

namespace nano
{
        function_sargan_t::function_sargan_t(const tensor_size_t dims) :
                function_t("Sargan", dims, 1, 100 * 1000, convexity::yes, 1e+6)
        {
        }

        scalar_t function_sargan_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                if (gx)
                {
                        *gx = scalar_t(1.2) * x.array() + scalar_t(0.8) * x.array().sum();
                }

                return  scalar_t(0.6) * x.array().square().sum() +
                        scalar_t(0.4) * nano::square(x.array().sum());
        }
}
