#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_color_cielab"

#include <boost/test/unit_test.hpp>
#include "cortex/util/logger.h"
#include "cortex/vision/color.h"

BOOST_AUTO_TEST_CASE(test_color_cielab)
{
        using namespace cortex;

        // test CIELab transform
        scalar_t min_cie_l = +1e100, min_cie_a = min_cie_l, min_cie_b = min_cie_l;
        scalar_t max_cie_l = -min_cie_l, max_cie_a = max_cie_l, max_cie_b = max_cie_l;

        for (rgba_t r = 0; r < 256; ++ r)
        {
                for (rgba_t g = 0; g < 256; g++)
                {
                        for (rgba_t b = 0; b < 256; ++ b)
                        {
                                const rgba_t rgba = color::make_rgba(r, g, b);
                                const cielab_t cielab = color::make_cielab(rgba);

                                BOOST_CHECK_EQUAL(rgba, color::make_rgba(cielab));

                                min_cie_l = std::min(min_cie_l, cielab(0));
                                min_cie_a = std::min(min_cie_a, cielab(1));
                                min_cie_b = std::min(min_cie_b, cielab(2));

                                max_cie_l = std::max(max_cie_l, cielab(0));
                                max_cie_a = std::max(max_cie_a, cielab(1));
                                max_cie_b = std::max(max_cie_b, cielab(2));
                        }
                }
        }

        log_info() << "CIELab range: L = [" << min_cie_l << ", " << max_cie_l
                        << "], a = [" << min_cie_a << ", " << max_cie_a
                        << "], b = [" << min_cie_b << ", " << max_cie_b << "].";
}
