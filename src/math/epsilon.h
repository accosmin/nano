#pragma once

#include <cmath>
#include <limits>

namespace nano
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
                return 10 * std::numeric_limits<tscalar>::epsilon();
        }

        template
        <
                typename tscalar
        >
        tscalar epsilon1()
        {
                return std::pow(std::cbrt(epsilon0<tscalar>()), tscalar(2));
        }

        template
        <
                typename tscalar
        >
        tscalar epsilon2()
        {
                return std::sqrt(epsilon0<tscalar>());
        }

        template
        <
                typename tscalar
        >
        tscalar epsilon3()
        {
                return std::cbrt(epsilon0<tscalar>());
        }
}

