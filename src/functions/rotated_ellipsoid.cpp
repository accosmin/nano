#include "math/numeric.h"
#include "rotated_ellipsoid.h"

namespace nano
{
        function_rotated_ellipsoid_t::function_rotated_ellipsoid_t(const tensor_size_t dims) :
                test_function_t("Rotated Hyper-Ellipsoid", dims, 1, 100 * 1000, convexity::yes, 100)
        {
        }

        scalar_t function_rotated_ellipsoid_t::vgrad(const vector_t& x, vector_t* gx) const
        {
                scalar_t fx = 0, fi = 0;
                for (auto i = 0; i < size(); i ++)
                {
                        fi += x(i);
                        fx += nano::square(fi);
                        if (gx)
                        {
                                (*gx)(i) = 2 * fi;
                        }
                }

                if (gx)
                {
                        for (auto i = size() - 2; i >= 0; i --)
                        {
                                (*gx)(i) += (*gx)(i + 1);
                        }
                }

                return fx;
        }
}
