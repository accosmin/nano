#include "rosenbrock.h"
#include "math/numeric.h"

namespace nano
{
        function_rosenbrock_t::function_rosenbrock_t(const tensor_size_t dims) :
                function_t("Rosenbrock", dims, 2, 100 * 1000, convexity::no, scalar_t(2.4))
        {
        }

        scalar_t function_rosenbrock_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                const auto ct = scalar_t(100);

                scalar_t fx = 0;
                for (tensor_size_t i = 0; i + 1 < size(); i ++)
                {
                        fx += ct * nano::square(x(i + 1) - x(i) * x(i)) + nano::square(x(i) - 1);
                }

                if (gx)
                {
                        (*gx).setZero();
                        for (tensor_size_t i = 0; i + 1 < size(); i ++)
                        {
                                (*gx)(i) += 2 * (x(i) - 1);
                                (*gx)(i) += ct * 2 * (x(i + 1) - x(i) * x(i)) * (- 2 * x(i));
                                (*gx)(i + 1) += ct * 2 * (x(i + 1) - x(i) * x(i));
                        }
                }

                return fx;
        }
}
