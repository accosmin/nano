//Link to Boost
#define BOOST_TEST_DYN_LINK

//Define our Module name (prints at testing)
#define BOOST_TEST_MODULE "BaseClassModule"

//VERY IMPORTANT - include this last
#include <boost/test/unit_test.hpp>

//#include "some_project/some_base_class.h"
// ------------- Tests Follow --------------

BOOST_AUTO_TEST_CASE(point)
{
        BOOST_CHECK(true);
        BOOST_CHECK_EQUAL(5, 5);
}
