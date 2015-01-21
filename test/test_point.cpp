#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE "test_nanocv"

#include <boost/test/unit_test.hpp>

#include "geom.h"

BOOST_AUTO_TEST_CASE(test_point)
{
        const ncv::point_t point(3, 7);

        BOOST_CHECK_EQUAL(point.x(), 3);
        BOOST_CHECK_EQUAL(point.y(), 7);
}
