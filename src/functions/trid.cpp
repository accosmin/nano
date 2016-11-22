#include "trid.h"

namespace nano
{
        function_trid_t::function_trid_t(const tensor_size_t dims) :
                test_function_t("Trid", dims, 2, 100 * 1000, convexity::yes, static_cast<scalar_t>(1 + dims * dims))
        {
        }

        scalar_t function_trid_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                if (gx)
                {
                        *gx = 2 * (x.array() - 1);
                        gx->segment(1, size() - 1) -= x.segment(0, size() - 1);
                        gx->segment(0, size() - 1) -= x.segment(1, size() - 1);
                }

                return (x.array() - 1).square().sum() -
                       (x.segment(0, size() - 1).array() * x.segment(1, size() - 1).array()).sum();
        }
}
