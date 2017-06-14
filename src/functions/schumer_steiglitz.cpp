#include "schumer_steiglitz.h"

using namespace nano;

function_schumer_steiglitz_t::function_schumer_steiglitz_t(const tensor_size_t dims) :
        function_t("Schumer-Steiglitz", dims, 1, 100 * 1000, convexity::yes, 1e+6)
{
}

scalar_t function_schumer_steiglitz_t::vgrad(const vector_t& x, vector_t* gx) const
{
        if (gx)
        {
                *gx = 4 * x.array().cube();
        }

        return x.array().square().square().sum();
}
