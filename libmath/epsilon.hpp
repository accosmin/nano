#pragma once

#include <cmath>
#include <limits>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief precision level [0=very precise, 1=quite precise, 2=precise, 3=loose] for different scalars
                ///
                template
                <
                        typename tscalar
                >
                tscalar epsilon0()
                {
                        return tscalar(10) * std::numeric_limits<tscalar>::epsilon();
                }

                template
                <
                        typename tscalar
                >
                tscalar epsilon1()
                {
                        const tscalar cbrt = std::cbrt(epsilon0<tscalar>());
                        return tscalar(2) * cbrt * cbrt;
                }

                template
                <
                        typename tscalar
                >
                tscalar epsilon2()
                {
                        return tscalar(2) * std::sqrt(epsilon0<tscalar>());
                }

                template
                <
                        typename tscalar
                >
                tscalar epsilon3()
                {
                        return tscalar(2) * std::cbrt(epsilon0<tscalar>());
                }
        }
}

