#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE "test_nanocv"

#include <boost/test/unit_test.hpp>

#include "geom.h"

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
        const ncv::rect_t rect(1, 2, 8, 6);

        BOOST_CHECK_EQUAL(rect.left(), 1);
        BOOST_CHECK_EQUAL(rect.top(), 2);
        BOOST_CHECK_EQUAL(rect.right(), 9);
        BOOST_CHECK_EQUAL(rect.bottom(), 8);

        BOOST_CHECK_EQUAL(rect.area(), 48);
        BOOST_CHECK_EQUAL(rect.empty(), false);
}
