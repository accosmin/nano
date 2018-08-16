#include "sargan.h"
#include "core/numeric.h"

using namespace nano;

function_sargan_t::function_sargan_t(const tensor_size_t dims) :
        function_t("Sargan", dims, 1, 100 * 1000, convexity::yes, 1e+6)
{
}

scalar_t function_sargan_t::vgrad(const vector_t& x, vector_t* gx) const
{
        const auto x2sum = x.dot(x);

        if (gx)
        {
                *gx = (scalar_t(1.2) + scalar_t(1.6) * x2sum) * x;
        }

        return scalar_t(0.6) * x2sum + scalar_t(0.4) * nano::square(x2sum);
}
