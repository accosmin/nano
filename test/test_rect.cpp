#include "unit_test.hpp"
#include "vision/rect.h"

namespace test
{
        void build_rect(cortex::coord_t x, cortex::coord_t y, cortex::coord_t w, cortex::coord_t h)
        {
                const cortex::rect_t rect(x, y, w, h);

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

ZOB_BEGIN_MODULE(test_rect)

ZOB_CASE(construction)
{
        test::build_rect(0, 0, 8, 6);
        test::build_rect(1, 2, 3, 4);
        test::build_rect(-1, -1, 7, 5);
}

ZOB_CASE(operations)
{
        using namespace cortex;

        // intersecting rectangles
        ZOB_CHECK_EQUAL(cortex::rect_t(1, 1, 3, 3) | cortex::rect_t(2, 2, 5, 4),
                           cortex::rect_t(1, 1, 6, 5));

        ZOB_CHECK_EQUAL(cortex::rect_t(1, 1, 3, 3) & cortex::rect_t(2, 2, 5, 4),
                           cortex::rect_t(2, 2, 2, 2));

        // disjoint rectangles
        ZOB_CHECK_EQUAL(cortex::rect_t(1, 1, 3, 3) & cortex::rect_t(7, 4, 5, 4),
                           cortex::rect_t(0, 0, 0, 0));

        // test center
        ZOB_CHECK_EQUAL(cortex::rect_t(1, 1, 3, 3).center(),
                           cortex::point_t(2, 2));
}

ZOB_END_MODULE()
