#pragma once

#include <limits>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief precision level [1=very precise, 2=precise, 3=loose] for different scalar types
                ///
                template
                <
                        typename tscalar
                >
                tscalar epsilon1()
                {
                        return tscalar(10) * std::numeric_limits<tscalar>::epsilon();
                }

                template
                <
                        typename tscalar
                >
                tscalar epsilon2()
                {
                        return tscalar(10) * std::sqrt(std::numeric_limits<tscalar>::epsilon());
                }

                template
                <
                        typename tscalar
                >
                tscalar epsilon3()
                {
                        return tscalar(10) * std::cbrt(std::numeric_limits<tscalar>::epsilon());
                }
        }
}

