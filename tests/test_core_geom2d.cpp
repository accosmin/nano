#include <utest/utest.h>
#include "core/geom2d.h"

UTEST_BEGIN_MODULE(test_core_geom2d)

UTEST_CASE(construct_point)
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

                UTEST_CHECK_EQUAL(point.x(), x);
                UTEST_CHECK_EQUAL(point.y(), y);
        }
}

UTEST_CASE(construct_rect)
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

                UTEST_CHECK_EQUAL(rect.left(),          x);
                UTEST_CHECK_EQUAL(rect.top(),           y);
                UTEST_CHECK_EQUAL(rect.right(),         x + w);
                UTEST_CHECK_EQUAL(rect.bottom(),        y + h);

                UTEST_CHECK_EQUAL(rect.area(),          w * h);
                UTEST_CHECK_EQUAL(rect.width(),         w);
                UTEST_CHECK_EQUAL(rect.height(),        h);
                UTEST_CHECK_EQUAL(rect.rows(),          h);
                UTEST_CHECK_EQUAL(rect.cols(),          w);

                // cppcheck-suppress compareBoolExpressionWithInt
                UTEST_CHECK_EQUAL(rect.valid(),         rect.area() > 0);
        }
}

UTEST_CASE(operations)
{
        // intersecting rectangles
        UTEST_CHECK_EQUAL(nano::rect_t(1, 1, 3, 3) | nano::rect_t(2, 2, 5, 4),
                        nano::rect_t(1, 1, 6, 5));

        UTEST_CHECK_EQUAL(nano::rect_t(1, 1, 3, 3) & nano::rect_t(2, 2, 5, 4),
                        nano::rect_t(2, 2, 2, 2));

        // disjoint rectangles
        UTEST_CHECK_EQUAL(nano::rect_t(1, 1, 3, 3) & nano::rect_t(7, 4, 5, 4),
                        nano::rect_t(0, 0, 0, 0));

        // test center
        UTEST_CHECK_EQUAL(nano::rect_t(1, 1, 3, 3).center(),
                        nano::point_t(2, 2));
}

UTEST_END_MODULE()
