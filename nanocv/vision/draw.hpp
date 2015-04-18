#pragma once

#include "rect.h"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/numeric.hpp"
#include <algorithm>

namespace ncv
{
        namespace draw_detail
        {
                template
                <
                        typename tmatrix,
                        typename toperator
                >
                bool apply(const rect_t& rect, const tmatrix& data, toperator op)
                {
                        const coord_t l = std::max(rect.left(), coord_t(0));
                        const coord_t r = std::min(rect.right(), static_cast<coord_t>(data.cols()));
                        const coord_t t = std::max(rect.top(), coord_t(0));
                        const coord_t b = std::min(rect.bottom(), static_cast<coord_t>(data.rows()));

                        for (coord_t x = l; x < r; x ++)
                        {
                                for (coord_t y = t; y < b; y ++)
                                {
                                        op(x, y, l, t, r, b);
                                }
                        }

                        return true;
                }
        }

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

                return draw_detail::apply(rect, data, [&] (coord_t x, coord_t y, coord_t, coord_t, coord_t, coord_t)
                {
                        if (math::square(x - cx) + math::square(y - cy) < math::square(radius))
                        {
                                data(y, x) = fill_value;
                        }
                });
        }

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

                const coord_t radiusx = (rect.width() + 1) / 2;
                const coord_t radiusy = (rect.height() + 1) / 2;

                return draw_detail::apply(rect, data, [&] (coord_t x, coord_t y, coord_t, coord_t, coord_t, coord_t)
                {
                        if (    math::square(x - cx) * math::square(radiusy) +
                                math::square(y - cy) * math::square(radiusx) <
                                math::square(radiusx * radiusy))
                        {
                                data(y, x) = fill_value;
                        }
                });
        }

        ///
        /// \brief draw a triangle in the given rectangle region
        ///
        template
        <
                typename tmatrix,
                typename tvalue
        >
        bool fill_up_triangle(const rect_t& rect, tmatrix& data, tvalue fill_value)
        {
                const coord_t w = rect.width(), w2 = (w + 1) / 2;
                const coord_t h = rect.height();

                return draw_detail::apply(rect, data, [&] (coord_t x, coord_t y, coord_t l, coord_t t, coord_t, coord_t)
                {
                        const coord_t dy = (h * math::abs(x - l - w2) + w2 - 1) / w2;

                        if (y - t >= dy)
                        {
                                data(y, x) = fill_value;
                        }
                });
        }

        ///
        /// \brief draw a triangle in the given rectangle region
        ///
        template
        <
                typename tmatrix,
                typename tvalue
        >
        bool fill_down_triangle(const rect_t& rect, tmatrix& data, tvalue fill_value)
        {
                const coord_t w = rect.width(), w2 = (w + 1) / 2;
                const coord_t h = rect.height();

                return draw_detail::apply(rect, data, [&] (coord_t x, coord_t y, coord_t l, coord_t t, coord_t r, coord_t)
                {
                        const coord_t dy = (h * (x < l + w2 ? x - l : r - x) + w2 - 1) / w2;

                        if (y - t <= dy)
                        {
                                data(y, x) = fill_value;
                        }
                });
        }
}
