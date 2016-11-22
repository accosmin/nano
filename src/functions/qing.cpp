#include "qing.h"

namespace nano
{
        function_qing_t::function_qing_t(const tensor_size_t dims) :
                test_function_t("Qing", dims, 2, 100 * 1000, convexity::no, static_cast<scalar_t>(dims))
        {
        }

        scalar_t function_qing_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const vector_t bias = vector_t::LinSpaced(size(), scalar_t(1), scalar_t(size()));

                if (gx)
                {
                        *gx = 4 * (x.array().square() - bias.array()) * x.array();
                }

                return (x.array().square() - bias.array()).square().sum();
        }
}
