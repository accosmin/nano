#include "utest.h"
#include "vision/rect.h"

namespace test
{
        void check_point(nano::coord_t x, nano::coord_t y)
        {
                const nano::point_t point(x, y);

                NANO_CHECK_EQUAL(point.x(), x);
                NANO_CHECK_EQUAL(point.y(), y);
        }

        void check_rect(nano::coord_t x, nano::coord_t y, nano::coord_t w, nano::coord_t h)
        {
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

NANO_BEGIN_MODULE(test_geom)

NANO_CASE(construct_point)
{
        test::check_point(3, 7);
        test::check_point(7, 3);
        test::check_point(-5, -1);
        test::check_point(-9, +1);
}

NANO_CASE(construct_rect)
{
        test::check_rect(0, 0, 8, 6);
        test::check_rect(1, 2, 3, 4);
        test::check_rect(-1, -1, 7, 5);
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
