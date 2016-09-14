#pragma once

#include <cmath>
#include <algorithm>

namespace nano
{
        ///
        /// \brief square
        ///
        template <typename tscalar>
        tscalar square(const tscalar value)
        {
                return value * value;
        }

        ///
        /// \brief cube
        ///
        template <typename tscalar>
        tscalar cube(const tscalar value)
        {
                return value * square(value);
        }

        ///
        /// \brief quartic
        ///
        template <typename tscalar>
        tscalar quartic(const tscalar value)
        {
                return square(square(value));
        }

        ///
        /// \brief integer division with rounding
        ///
        template <typename tinteger, typename tinteger2>
        tinteger idiv(const tinteger nominator, const tinteger2 denominator)
        {
                return (nominator + static_cast<tinteger>(denominator) - 1) / static_cast<tinteger>(denominator);
        }

        ///
        /// \brief absolute value
        ///
        template <typename tscalar>
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

        ///
        /// \brief check if two scalars are almost equal
        ///
        template <typename tscalar>
        bool close(const tscalar x, const tscalar y, const tscalar epsilon)
        {
                return nano::abs(x - y) <= (tscalar(1) + std::max(x, y)) * epsilon;
        }

        ///
        /// \brief clamp value in the [min_value, max_value] range
        ///
        template
        <
                typename tscalar,
                typename tscalar_min,
                typename tscalar_max
        >
        tscalar clamp(tscalar value, tscalar_min min_value, tscalar_max max_value)
        {
                return  value < static_cast<tscalar>(min_value) ? static_cast<tscalar>(min_value) :
                        (value > static_cast<tscalar>(max_value) ? static_cast<tscalar>(max_value) : value);
        }
}

