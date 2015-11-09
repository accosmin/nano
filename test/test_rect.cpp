#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_rect"

#include <boost/test/unit_test.hpp>
#include "cortex/vision/rect.h"

namespace test
{
        void build_rect(cortex::coord_t x, cortex::coord_t y, cortex::coord_t w, cortex::coord_t h)
        {
                const cortex::rect_t rect(x, y, w, h);

                BOOST_CHECK_EQUAL(rect.left(),          x);
                BOOST_CHECK_EQUAL(rect.top(),           y);
                BOOST_CHECK_EQUAL(rect.right(),         x + w);
                BOOST_CHECK_EQUAL(rect.bottom(),        y + h);

                BOOST_CHECK_EQUAL(rect.area(),          w * h);
                BOOST_CHECK_EQUAL(rect.width(),         w);
                BOOST_CHECK_EQUAL(rect.height(),        h);
                BOOST_CHECK_EQUAL(rect.rows(),          h);
                BOOST_CHECK_EQUAL(rect.cols(),          w);

                BOOST_CHECK_EQUAL(rect.valid(),         rect.area() > 0);
        }
}

BOOST_AUTO_TEST_CASE(test_rect_construction)
{
        test::build_rect(0, 0, 8, 6);
        test::build_rect(1, 2, 3, 4);
        test::build_rect(-1, -1, 7, 5);
}

BOOST_AUTO_TEST_CASE(test_rect_operations)
{
        using namespace cortex;

        // intersecting rectangles
        BOOST_CHECK_EQUAL(cortex::rect_t(1, 1, 3, 3) | cortex::rect_t(2, 2, 5, 4),
                          cortex::rect_t(1, 1, 6, 5));

        BOOST_CHECK_EQUAL(cortex::rect_t(1, 1, 3, 3) & cortex::rect_t(2, 2, 5, 4),
                          cortex::rect_t(2, 2, 2, 2));

        // disjoint rectangles
        BOOST_CHECK_EQUAL(cortex::rect_t(1, 1, 3, 3) & cortex::rect_t(7, 4, 5, 4),
                          cortex::rect_t(0, 0, 0, 0));

        // test center
        BOOST_CHECK_EQUAL(cortex::rect_t(1, 1, 3, 3).center(),
                          cortex::point_t(2, 2));
}
