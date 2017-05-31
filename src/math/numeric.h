#pragma once

#include <cmath>
#include <algorithm>

namespace nano
{
        ///
        /// \brief square: x^2
        ///
        template <typename tscalar>
        tscalar square(const tscalar value)
        {
                return value * value;
        }

        ///
        /// \brief cube: x^3
        ///
        template <typename tscalar>
        tscalar cube(const tscalar value)
        {
                return value * square(value);
        }

        ///
        /// \brief quartic: x^4
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
        /// \brief integer rounding
        ///
        template <typename tinteger, typename tinteger2>
        tinteger iround(const tinteger value, const tinteger2 modulo)
        {
                return idiv(value, modulo) * modulo;
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
        /// \todo replace this with std::clamp when moving to C++17
        ///
        template
        <
                typename tscalar,
                typename tscalar_min,
                typename tscalar_max
        >
        tscalar clamp(const tscalar value, const tscalar_min min_value, const tscalar_max max_value)
        {
                return  value < static_cast<tscalar>(min_value) ? static_cast<tscalar>(min_value) :
                        (value > static_cast<tscalar>(max_value) ? static_cast<tscalar>(max_value) : value);
        }
}

