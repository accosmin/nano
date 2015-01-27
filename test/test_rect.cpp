#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_rect"

#include <boost/test/unit_test.hpp>
#include "libnanocv/rect.h"

namespace test
{
        void build_rect(ncv::coord_t x, ncv::coord_t y, ncv::coord_t w, ncv::coord_t h)
        {
                const ncv::rect_t rect(x, y, w, h);

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
        using namespace ncv;

        // intersecting rectangles
        BOOST_CHECK_EQUAL(ncv::rect_t(1, 1, 3, 3) | ncv::rect_t(2, 2, 5, 4),
                          ncv::rect_t(1, 1, 6, 5));

        BOOST_CHECK_EQUAL(ncv::rect_t(1, 1, 3, 3) & ncv::rect_t(2, 2, 5, 4),
                          ncv::rect_t(2, 2, 2, 2));

        // disjoint rectangles
        BOOST_CHECK_EQUAL(ncv::rect_t(1, 1, 3, 3) & ncv::rect_t(7, 4, 5, 4),
                          ncv::rect_t(0, 0, 0, 0));

        // test center
        BOOST_CHECK_EQUAL(ncv::rect_t(1, 1, 3, 3).center(),
                          ncv::point_t(2, 2));
}
