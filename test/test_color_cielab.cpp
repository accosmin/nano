#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_color_cielab"

#include <boost/test/unit_test.hpp>
#include "cortex/vision/color.h"
#include "math/random.hpp"

BOOST_AUTO_TEST_CASE(test_color_cielab)
{
        using namespace cortex;

        const size_t tests = 1023;

        math::random_t<rgba_t> rgen(0, 255);
        math::random_t<rgba_t> ggen(0, 255);
        math::random_t<rgba_t> bgen(0, 255);

        for (size_t t = 0; t < tests; ++ t)
        {
                const rgba_t r = rgen();
                const rgba_t g = ggen();
                const rgba_t b = bgen();

                const rgba_t rgba = color::make_rgba(r, g, b);
                const cielab_t cielab = color::make_cielab(rgba);

                BOOST_CHECK_EQUAL(rgba, color::make_rgba(cielab));
                BOOST_CHECK_GE(cielab(0), 0.0);
                BOOST_CHECK_LE(cielab(0), 100.0);
        }
}
