#include "geom.h"
#include <boost/geometry.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        rect_t geom::intersection(const rect_t& rect1, const rect_t& rect2)
        {
                rect_t result;
                if (boost::geometry::intersection(rect1, rect2, result))
                {
                        return result;
                }
                else
                {
                        return make_rect(0, 0, 0, 0);
                }
        }

        //-------------------------------------------------------------------------------------------------

        rect_t geom::union_(const rect_t& rect1, const rect_t& rect2)
        {
                const coord_t l = std::min(left(rect1), left(rect2));
                const coord_t r = std::max(right(rect1), right(rect2));
                const coord_t t = std::min(top(rect1), top(rect2));
                const coord_t b = std::max(bottom(rect1), bottom(rect2));

                return make_rect(l, t, r - l, b - t);
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t geom::overlap(const rect_t& rect1, const rect_t& rect2)
        {
                return (area(intersection(rect1, rect2)) + 1.0) /
                       (area(union_(rect1, rect2)) + 1.0);
        }

        //-------------------------------------------------------------------------------------------------
}


