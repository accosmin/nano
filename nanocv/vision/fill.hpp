#pragma once

#include "rect.h"
#include <algorithm>

namespace ncv
{
        namespace fill_detail
        {
                template
                <
                        typename tmatrix,
                        typename toperator
                >
                bool apply(const rect_t& rect, const tmatrix& data, const toperator& op)
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
}
