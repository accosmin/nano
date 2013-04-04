#include "ncv.h"

int main(int argc, char *argv[])
{
        ncv::log_info() << "testing RGBA transform ...";

        // test RGBA transform
        for (ncv::rgba_t r = 0; r < 256; r ++)
        {
                for (ncv::rgba_t g = 0; g < 256; g++)
                {
                        for (ncv::rgba_t b = 0; b < 256; b ++)
                        {
                                const ncv::rgba_t rgba = ncv::color::make_rgba(r, g, b);
                                if (    ncv::color::rgba2r(rgba) != r ||
                                        ncv::color::rgba2g(rgba) != g ||
                                        ncv::color::rgba2b(rgba) != b)
                                {
                                        ncv::log_error() << "failed!";
                                        return EXIT_FAILURE;
                                }
                        }
                }
        }

        ncv::log_info() << ">>> done.";

        ncv::log_info() << "testing CIELab transform ...";

        // test CIELab transform
        ncv::scalar_t min_cie_l = +1e100, min_cie_a = min_cie_l, min_cie_b = min_cie_l;
        ncv::scalar_t max_cie_l = -min_cie_l, max_cie_a = max_cie_l, max_cie_b = max_cie_l;

        for (ncv::rgba_t r = 0; r < 256; r ++)
        {
                for (ncv::rgba_t g = 0; g < 256; g++)
                {
                        for (ncv::rgba_t b = 0; b < 256; b ++)
                        {
                                ncv::scalar_t cie_l, cie_a, cie_b;
                                ncv::rgba_t rgb_r = r, rgb_g = g, rgb_b = b;

                                ncv::color::rgb2lab(rgb_r, rgb_g, rgb_b, cie_l, cie_a, cie_b);
                                ncv::color::lab2rgb(cie_l, cie_a, cie_b, rgb_r, rgb_g, rgb_b);

                                if (    rgb_r != r ||
                                        rgb_g != g ||
                                        rgb_b != b)
                                {
                                        ncv::log_error() << "failed!";
                                        return EXIT_FAILURE;
                                }

                                min_cie_l = std::min(min_cie_l, cie_l);
                                min_cie_a = std::min(min_cie_a, cie_a);
                                min_cie_b = std::min(min_cie_b, cie_b);

                                max_cie_l = std::max(max_cie_l, cie_l);
                                max_cie_a = std::max(max_cie_a, cie_a);
                                max_cie_b = std::max(max_cie_b, cie_b);
                        }
                }
        }

        ncv::log_info() << ">>> done.";

        ncv::log_info() << "CIELab range: L = [" << min_cie_l << ", " << max_cie_l
                        << "], a = [" << min_cie_a << ", " << max_cie_a
                        << "], b = [" << min_cie_b << ", " << max_cie_b << "].";

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
