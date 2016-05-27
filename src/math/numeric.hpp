#pragma once

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
}

