#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_color_rgba"

#include <boost/test/unit_test.hpp>
#include "nanocv/color.h"
#include "util/random.hpp"

BOOST_AUTO_TEST_CASE(test_color_rgba)
{
        using namespace ncv;

        const size_t tests = 256 * 1024;

        ncv::random_t<rgba_t> rgen(0, 255);
        ncv::random_t<rgba_t> ggen(0, 255);
        ncv::random_t<rgba_t> bgen(0, 255);
        ncv::random_t<rgba_t> agen(0, 255);

        // test RGBA transform
        for (size_t t = 0; t < tests; t ++)
        {
                const rgba_t r = rgen();
                const rgba_t g = ggen();
                const rgba_t b = bgen();
                const rgba_t a = agen();

                const rgba_t rgba = color::make_rgba(r, g, b, a);

                BOOST_CHECK_EQUAL(color::make_luma(rgba), color::make_luma(r, g, b));

                BOOST_CHECK_EQUAL(r, color::get_red(rgba));
                BOOST_CHECK_EQUAL(g, color::get_green(rgba));
                BOOST_CHECK_EQUAL(b, color::get_blue(rgba));
                BOOST_CHECK_EQUAL(a, color::get_alpha(rgba));

                BOOST_CHECK_EQUAL(rgba, color::set_red(rgba, r));
                BOOST_CHECK_EQUAL(rgba, color::set_green(rgba, g));
                BOOST_CHECK_EQUAL(rgba, color::set_blue(rgba, b));
                BOOST_CHECK_EQUAL(rgba, color::set_alpha(rgba, a));
        }
}
