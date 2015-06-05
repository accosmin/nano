#pragma once

#include "fill.hpp"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/numeric.hpp"

namespace ncv
{
        ///
        /// \brief draw a circle in the given rectangle region
        ///
        template
        <
                typename tmatrix,
                typename tvalue
        >
        bool fill_circle(const rect_t& rect, tmatrix& data, tvalue fill_value)
        {
                const point_t center = rect.center();
                const coord_t cx = center.x();
                const coord_t cy = center.y();

                const coord_t radius = (std::min(rect.width(), rect.height()) + 1) / 2;

                return fill_detail::apply(rect, data, [&] (coord_t x, coord_t y, coord_t, coord_t, coord_t, coord_t)
                {
                        if (math::square(x - cx) + math::square(y - cy) < math::square(radius))
                        {
                                data(y, x) = fill_value;
                        }
                });
        }
}
