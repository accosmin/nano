#include "geom.h"
#include <algorithm>

namespace ncv
{
        rect_t rect_t::intersection(const rect_t& other) const
        {
                const coord_t xmin1 = left(), xmin2 = other.left();
                const coord_t xmax1 = right(), xmax2 = other.right();

                const coord_t ymin1 = top(), ymin2 = other.top();
                const coord_t ymax1 = bottom(), ymax2 = other.bottom();

                if (    xmax1 < xmin2 || xmax2 < xmin1 ||
                        ymax1 < ymin2 || ymax2 < ymin1)
                {
                        return rect_t(0, 0, 0, 0);
                }

                else
                {
                        const coord_t xmin = xmin1 < xmin2 ? xmin2 : xmin1;
                        const coord_t xmax = xmax1 > xmax2 ? xmax2 : xmax1;

                        const coord_t ymin = ymin1 < ymin2 ? ymin2 : ymin1;
                        const coord_t ymax = ymax1 > ymax2 ? ymax2 : ymax1;

                        return rect_t(xmin, ymin, xmax - xmin, ymax - ymin);
                }
        }

        rect_t rect_t::union_(const rect_t& other) const
        {
                const coord_t l = std::min(left(), other.left());
                const coord_t r = std::max(right(), other.right());
                const coord_t t = std::min(top(), other.top());
                const coord_t b = std::max(bottom(), other.bottom());

                return rect_t(l, t, r - l, b - t);
        }

        scalar_t rect_t::overlap(const rect_t& other) const
        {
                return (intersection(other).area() + 1.0) /
                       (union_(other).area() + 1.0);
        }
}


