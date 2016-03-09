#include "unit_test.hpp"
#include "vision/rect.h"

namespace test
{
        void check_point(zob::coord_t x, zob::coord_t y)
        {
                const zob::point_t point(x, y);

                ZOB_CHECK_EQUAL(point.x(), x);
                ZOB_CHECK_EQUAL(point.y(), y);
        }

        void check_rect(zob::coord_t x, zob::coord_t y, zob::coord_t w, zob::coord_t h)
        {
                const zob::rect_t rect(x, y, w, h);

                ZOB_CHECK_EQUAL(rect.left(),          x);
                ZOB_CHECK_EQUAL(rect.top(),           y);
                ZOB_CHECK_EQUAL(rect.right(),         x + w);
                ZOB_CHECK_EQUAL(rect.bottom(),        y + h);

                ZOB_CHECK_EQUAL(rect.area(),          w * h);
                ZOB_CHECK_EQUAL(rect.width(),         w);
                ZOB_CHECK_EQUAL(rect.height(),        h);
                ZOB_CHECK_EQUAL(rect.rows(),          h);
                ZOB_CHECK_EQUAL(rect.cols(),          w);

                ZOB_CHECK_EQUAL(rect.valid(),         rect.area() > 0);
        }
}

ZOB_BEGIN_MODULE(test_geom)

ZOB_CASE(construct_point)
{
        test::check_point(3, 7);
        test::check_point(7, 3);
        test::check_point(-5, -1);
        test::check_point(-9, +1);
}

ZOB_CASE(construct_rect)
{
        test::check_rect(0, 0, 8, 6);
        test::check_rect(1, 2, 3, 4);
        test::check_rect(-1, -1, 7, 5);
}

ZOB_CASE(operations)
{
        // intersecting rectangles
        ZOB_CHECK_EQUAL(zob::rect_t(1, 1, 3, 3) | zob::rect_t(2, 2, 5, 4),
                        zob::rect_t(1, 1, 6, 5));

        ZOB_CHECK_EQUAL(zob::rect_t(1, 1, 3, 3) & zob::rect_t(2, 2, 5, 4),
                        zob::rect_t(2, 2, 2, 2));

        // disjoint rectangles
        ZOB_CHECK_EQUAL(zob::rect_t(1, 1, 3, 3) & zob::rect_t(7, 4, 5, 4),
                        zob::rect_t(0, 0, 0, 0));

        // test center
        ZOB_CHECK_EQUAL(zob::rect_t(1, 1, 3, 3).center(),
                        zob::point_t(2, 2));
}

ZOB_END_MODULE()
