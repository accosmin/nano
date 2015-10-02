#pragma once

#include "abs.hpp"

namespace math
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
                return math::abs(x - y) <= (tscalar(1) + std::max(x, y)) * epsilon;
        }
}

