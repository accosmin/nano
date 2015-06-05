#pragma once

#include "fill.hpp"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/numeric.hpp"

namespace ncv
{
        ///
        /// \brief draw an ellipse in the given rectangle region
        ///
        template
        <
                typename tmatrix,
                typename tvalue
        >
        bool fill_ellipse(const rect_t& rect, tmatrix& data, tvalue fill_value)
        {
                const point_t center = rect.center();
                const coord_t cx = center.x();
                const coord_t cy = center.y();

                const coord_t radx = (rect.width() + 1) / 2;
                const coord_t rady = (rect.height() + 1) / 2;

                return  fill_detail::apply(rect, data,
                        [&data = data, fill_value = fill_value, cx = cx, cy = cy, radx = radx, rady = rady]
                        (coord_t x, coord_t y, coord_t, coord_t, coord_t, coord_t)
                {
                        if (    math::square(x - cx) * math::square(rady) +
                                math::square(y - cy) * math::square(radx) <
                                math::square(radx * rady))
                        {
                                data(y, x) = fill_value;
                        }
                });
        }
}
