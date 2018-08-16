#include "cauchy.h"

using namespace nano;

function_cauchy_t::function_cauchy_t(const tensor_size_t dims) :
        function_t("Cauchy", dims, 1, 100 * 1000, convexity::yes, 1)
{
}

scalar_t function_cauchy_t::vgrad(const vector_t& x, vector_t* gx) const
{
        if (gx)
        {
                *gx = 2 * x / (1 + x.dot(x));
        }

        return std::log1p(x.dot(x));
}
