#pragma once

#include <cmath>
#include <limits>

namespace nano
{
        ///
        /// \brief round to the closest power of 10
        ///
        template <typename tscalar>
        inline auto roundpow10(const tscalar v)
        {
                return std::pow(tscalar(10), std::floor(std::log10(v)));
        }

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
                return 10 * roundpow10(epsilon<tscalar>());
        }

        template <typename tscalar>
        tscalar epsilon1()
        {
                const auto cb = std::cbrt(epsilon<tscalar>());
                return 20 * roundpow10(cb * cb);
        }

        template <typename tscalar>
        tscalar epsilon2()
        {
                return 20 * roundpow10(std::sqrt(epsilon<tscalar>()));
        }

        template <typename tscalar>
        tscalar epsilon3()
        {
                return 20 * roundpow10(std::cbrt(epsilon<tscalar>()));
        }
}
