#include "exponential.h"

using namespace nano;

function_exponential_t::function_exponential_t(const tensor_size_t dims) :
        function_t("Exponential", dims, 1, 100 * 1000, convexity::yes, 1)
{
}

scalar_t function_exponential_t::vgrad(const vector_t& x, vector_t* gx) const
{
        const auto fx = std::exp(1 + x.dot(x) / scalar_t(size()));

        if (gx)
        {
                *gx = (2 * fx / scalar_t(size())) * x;
        };

        return fx;
}
