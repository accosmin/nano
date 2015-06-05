#pragma once

#include "fill.hpp"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/numeric.hpp"

namespace ncv
{
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

                return  fill_detail::apply(rect, data,
                        [&data = data, fill_value = fill_value, h = h, w2 = w2]
                        (coord_t x, coord_t y, coord_t l, coord_t t, coord_t, coord_t)
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

                return  fill_detail::apply(rect, data,
                        [&data = data, fill_value = fill_value, h = h, w2 = w2]
                        (coord_t x, coord_t y, coord_t l, coord_t t, coord_t r, coord_t)
                {
                        const coord_t dy = (h * (x < l + w2 ? x - l : r - x) + w2 - 1) / w2;

                        if (y - t <= dy)
                        {
                                data(y, x) = fill_value;
                        }
                });
        }
}
