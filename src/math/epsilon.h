#pragma once

#include <cmath>
#include <limits>

namespace nano
{
        ///
        /// \brief precision level [0=very precise, 1=quite precise, 2=precise, 3=loose] for different scalars
        ///
        template <typename tscalar>
        tscalar epsilon()
        {
                return std::numeric_limits<tscalar>::epsilon();
        }

        template <typename tscalar>
        tscalar epsilon0()
        {
                return 10 * epsilon<tscalar>();
        }

        template <typename tscalar>
        tscalar epsilon1()
        {
                const auto cb = std::cbrt(epsilon<tscalar>());
                return 2 * cb * cb;
        }

        template <typename tscalar>
        tscalar epsilon2()
        {
                return 2 * std::sqrt(epsilon<tscalar>());
        }

        template <typename tscalar>
        tscalar epsilon3()
        {
                return 2 * std::cbrt(epsilon<tscalar>());
        }
}
