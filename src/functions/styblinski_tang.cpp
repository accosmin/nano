#include "styblinski_tang.h"

using namespace nano;

function_styblinski_tang_t::function_styblinski_tang_t(const tensor_size_t dims) :
        function_t("Styblinski-Tang", dims, 1, 100 * 1000, convexity::no, 5)
{
}

scalar_t function_styblinski_tang_t::vgrad(const vector_t& x, vector_t* gx) const
{
        if (gx)
        {
                *gx = 4 * x.array().cube() - 32 * x.array() + 5;
        }

        return (x.array().square().square() - 16 * x.array().square() + 5 * x.array()).sum();
}
