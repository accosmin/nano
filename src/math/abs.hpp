#pragma once

#include <cmath>

namespace nano
{
        ///
        /// \brief absolute value
        ///
        template
        <
                typename tscalar
        >
        tscalar abs(tscalar v)
        {
                return std::abs(v);
        }

        template <>
        inline float abs(float v)
        {
                return std::fabs(v);
        }

        template <>
        inline double abs(double v)
        {
                return std::fabs(v);
        }

        template <>
        inline long double abs(long double v)
        {
                return std::fabs(v);
        }
}

