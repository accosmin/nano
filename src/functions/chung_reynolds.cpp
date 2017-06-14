#include "chung_reynolds.h"

using namespace nano;

function_chung_reynolds_t::function_chung_reynolds_t(const tensor_size_t dims) :
        function_t("Chung-Reynolds", dims, 1, 100 * 1000, convexity::yes, 1)
{
}

scalar_t function_chung_reynolds_t::vgrad(const vector_t& x, vector_t* gx) const
{
        const auto u = x.array().square().sum();

        if (gx)
        {
                *gx = (4 * u) * x;
        }

        return u * u;
}
