#include "exponential.h"

namespace nano
{
        function_exponential_t::function_exponential_t(const tensor_size_t dims) :
                function_t("Exponential", dims, 1, 100 * 1000, convexity::yes, 1)
        {
        }

        scalar_t function_exponential_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto fx = std::exp(scalar_t(0.5) / scalar_t(size()) * x.array().square().sum());

                if (gx)
                {
                        *gx = fx * x / scalar_t(size());
                };

                return fx;
        }
}
