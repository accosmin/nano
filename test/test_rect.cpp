#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE "test_nanocv"

#include <boost/test/unit_test.hpp>

#include "geom.h"

BOOST_AUTO_TEST_CASE(test_rect_construction)
{
        const ncv::rect_t rect(1, 2, 8, 6);

        BOOST_CHECK_EQUAL(rect.left(), 1);
        BOOST_CHECK_EQUAL(rect.top(), 2);
        BOOST_CHECK_EQUAL(rect.right(), 9);
        BOOST_CHECK_EQUAL(rect.bottom(), 8);
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
