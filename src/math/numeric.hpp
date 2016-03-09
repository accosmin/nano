#pragma once

namespace zob
{
        ///
        /// \brief square
        ///
        template
        <
                typename tvalue
        >
        tvalue square(tvalue value)
        {
                return value * value;
        }

        ///
        /// \brief cube
        ///
        template
        <
                typename tvalue
        >
        tvalue cube(tvalue value)
        {
                return value * square(value);
        }

        ///
        /// \brief quartic
        ///
        template
        <
                typename tvalue
        >
        tvalue quartic(tvalue value)
        {
                return square(square(value));
        }
}

