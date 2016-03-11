#pragma once

#include "abs.hpp"

namespace nano
{
        ///
        /// \brief check if two scalars are almost equal
        ///
        template
        <
                typename tscalar
        >
        bool close(tscalar x, tscalar y, tscalar epsilon)
        {
                return nano::abs(x - y) <= (tscalar(1) + std::max(x, y)) * epsilon;
        }
}

