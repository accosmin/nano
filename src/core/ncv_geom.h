#ifndef NANOCV_GEOM_H
#define NANOCV_GEOM_H

#include "ncv_types.h"

namespace ncv
{
        namespace geom
        {
                // create geometric objects
                inline point_t make_point(coord_t x = 0, coord_t y = 0)
                {
                        return point_t(x, y);
                }
                inline rect_t make_rect(coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0)
                {
                        return rect_t(make_point(x, y), make_point(x + w, y + h));
                }
                inline rect_t make_size(coord_t w = 0, coord_t h = 0)
                {
                        return make_rect(0, 0, w, h);
                }

                // access geometric objects
                inline coord_t left(const rect_t& rect)     { return rect.min_corner().x(); }
                inline coord_t right(const rect_t& rect)    { return rect.max_corner().x(); }
                inline coord_t top(const rect_t& rect)      { return rect.min_corner().y(); }
                inline coord_t bottom(const rect_t& rect)   { return rect.max_corner().y(); }

                inline coord_t width(const rect_t& rect)    { return right(rect) - left(rect); }
                inline coord_t height(const rect_t& rect)   { return bottom(rect) - top(rect); }
                inline coord_t rows(const rect_t& rect)     { return height(rect); }
                inline coord_t cols(const rect_t& rect)     { return width(rect); }

                inline coord_t area(const rect_t& rect)     { return width(rect) * height(rect); }

                // compute the overlap between two regions
                rect_t intersection(const rect_t& rect1, const rect_t& rect2);
                rect_t union_(const rect_t& rect1, const rect_t& rect2);
                scalar_t overlap(const rect_t& rect1, const rect_t& rect2);
        }
}

#endif // NANOCV_GEOM_H

