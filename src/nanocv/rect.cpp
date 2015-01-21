#include "rect.h"
#include <algorithm>

namespace ncv
{
        rect_t operator&(const rect_t& rect1, const rect_t& rect2)
        {
                const coord_t top = std::max(rect1.top(), rect2.top());
                const coord_t left = std::max(rect1.left(), rect2.left());

                const coord_t right = std::min(rect1.right(), rect2.right());                
                const coord_t bottom = std::min(rect1.bottom(), rect2.bottom());

                if (right >= left && bottom >= top)
                {
                        return rect_t(left, top, right - left, bottom - top);
                }
                else
                {
                        return rect_t(0, 0, 0, 0);
                }
        }

        rect_t operator|(const rect_t& rect1, const rect_t& rect2)
        {
                const coord_t top = std::min(rect1.top(), rect2.top());
                const coord_t left = std::min(rect1.left(), rect2.left());

                const coord_t right = std::max(rect1.right(), rect2.right());
                const coord_t bottom = std::max(rect1.bottom(), rect2.bottom());

                return rect_t(left, top, right - left, bottom - top);
        }

        scalar_t overlap(const rect_t& rect1, const rect_t& rect2)
        {
                return ((rect1 & rect2).area() + 1.0) /
                       ((rect1 | rect2).area() + 1.0);
        }
}


