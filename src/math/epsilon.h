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
                return 10 * std::pow(std::cbrt(std::numeric_limits<tscalar>::epsilon()), tscalar(2));
        }

        template
        <
                typename tscalar
        >
        tscalar epsilon2()
        {
                return 10 * std::sqrt(std::numeric_limits<tscalar>::epsilon());
        }

        template
        <
                typename tscalar
        >
        tscalar epsilon3()
        {
                return 10 * std::cbrt(std::numeric_limits<tscalar>::epsilon());
        }
}

