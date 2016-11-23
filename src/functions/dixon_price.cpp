#include "dixon_price.h"
#include "math/numeric.h"

namespace nano
{
        function_dixon_price_t::function_dixon_price_t(const tensor_size_t dims) :
                function_t("Dixon-Price", dims, 2, 100 * 1000, convexity::no, 10)
        {
        }

        scalar_t function_dixon_price_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const vector_t bias = vector_t::LinSpaced(size(), scalar_t(1), scalar_t(size()));

                const auto xsegm0 = x.segment(0, size() - 1);
                const auto xsegm1 = x.segment(1, size() - 1);

                if (gx)
                {
                        const auto weight = bias.segment(1, size() - 1).array() *
                                2 * (2 * xsegm1.array().square() - xsegm0.array());

                        (*gx).setZero();
                        (*gx)(0) = 2 * (x(0) - 1);
                        (*gx).segment(1, size() - 1).array() += weight * 4 * xsegm1.array();
                        (*gx).segment(0, size() - 1).array() -= weight;
                }

                return  nano::square(x(0) - 1) +
                        (bias.segment(1, size() - 1).array() *
                        (2 * xsegm1.array().square() - xsegm0.array()).square()).sum();
        }
}
