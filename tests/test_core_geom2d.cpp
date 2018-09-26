#include "utest.h"
#include "core/geom2d.h"

NANO_BEGIN_MODULE(test_core_geom2d)

NANO_CASE(construct_point)
{
        for (const auto& xy : {
                std::make_tuple<nano::coord_t>(+3, +7),
                std::make_tuple<nano::coord_t>(+7, +3),
                std::make_tuple<nano::coord_t>(-5, -1),
                std::make_tuple<nano::coord_t>(-9, +1)})
        {
                const auto x = std::get<0>(xy);
                const auto y = std::get<1>(xy);

                const nano::point_t point(x, y);

                NANO_CHECK_EQUAL(point.x(), x);
                NANO_CHECK_EQUAL(point.y(), y);
        }
}

NANO_CASE(construct_rect)
{
        for (const auto& xywh : {
                std::make_tuple<nano::coord_t>(0, 0, 8, 6),
                std::make_tuple<nano::coord_t>(1, 2, 3, 4),
                std::make_tuple<nano::coord_t>(-1, -1, 7, 5)})
        {
                const auto x = std::get<0>(xywh);
                const auto y = std::get<1>(xywh);
                const auto w = std::get<2>(xywh);
                const auto h = std::get<3>(xywh);

                const nano::rect_t rect(x, y, w, h);

                NANO_CHECK_EQUAL(rect.left(),          x);
                NANO_CHECK_EQUAL(rect.top(),           y);
                NANO_CHECK_EQUAL(rect.right(),         x + w);
                NANO_CHECK_EQUAL(rect.bottom(),        y + h);

                NANO_CHECK_EQUAL(rect.area(),          w * h);
                NANO_CHECK_EQUAL(rect.width(),         w);
                NANO_CHECK_EQUAL(rect.height(),        h);
                NANO_CHECK_EQUAL(rect.rows(),          h);
                NANO_CHECK_EQUAL(rect.cols(),          w);

                NANO_CHECK_EQUAL(rect.valid(),         rect.area() > 0);
        }
}

NANO_CASE(operations)
{
        // intersecting rectangles
        NANO_CHECK_EQUAL(nano::rect_t(1, 1, 3, 3) | nano::rect_t(2, 2, 5, 4),
                        nano::rect_t(1, 1, 6, 5));

        NANO_CHECK_EQUAL(nano::rect_t(1, 1, 3, 3) & nano::rect_t(2, 2, 5, 4),
                        nano::rect_t(2, 2, 2, 2));

        // disjoint rectangles
        NANO_CHECK_EQUAL(nano::rect_t(1, 1, 3, 3) & nano::rect_t(7, 4, 5, 4),
                        nano::rect_t(0, 0, 0, 0));

        // test center
        NANO_CHECK_EQUAL(nano::rect_t(1, 1, 3, 3).center(),
                        nano::point_t(2, 2));
}

NANO_END_MODULE()
