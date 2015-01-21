#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_nanocv"

#include <boost/test/unit_test.hpp>
#include "nanocv/color.h"

BOOST_AUTO_TEST_CASE(test_color_rgba)
{
        using namespace ncv;

        // test RGBA transform
        for (rgba_t r = 0; r < 256; r ++)
        {
                for (rgba_t g = 0; g < 256; g ++)
                {
                        for (rgba_t b = 0; b < 256; b ++)
                        {
                                for (rgba_t a = 0; a < 256; a ++)
                                {
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
                }
        }
}
