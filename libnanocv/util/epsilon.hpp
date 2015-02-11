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
                        return tscalar(0);
                }

                template <>
                inline float epsilon1<float>()
                {
                        return std::numeric_limits<float>::epsilon();
                }

                template <>
                inline double epsilon1<double>()
                {
                        return 1e+3 * std::numeric_limits<double>::epsilon();
                }

                template <>
                inline long double epsilon1<long double>()
                {
                        return 1e+5L * std::numeric_limits<long double>::epsilon();
                }

                template
                <
                        typename tscalar
                >
                tscalar epsilon2()
                {
                        return static_cast<tscalar>(std::sqrt(epsilon1<tscalar>()));
                }

                template
                <
                        typename tscalar
                >
                tscalar epsilon3()
                {
                        return static_cast<tscalar>(std::cbrt(epsilon1<tscalar>()));
                }
        }
}

