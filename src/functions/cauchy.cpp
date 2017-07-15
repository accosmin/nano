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
                *gx = (2 * x.array()) / (1 + x.array().square());
        }

        return (1 + x.array().square()).log().sum();
}
