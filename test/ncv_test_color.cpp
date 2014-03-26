#include "nanocv.h"

int main(int argc, char *argv[])
{
        // test RGBA transform
        {
                ncv::log_info() << "testing RGBA transform ...";

                for (ncv::rgba_t r = 0; r < 256; r ++)
                {
                        for (ncv::rgba_t g = 0; g < 256; g++)
                        {
                                for (ncv::rgba_t b = 0; b < 256; b ++)
                                {
                                        const ncv::rgba_t rgba = ncv::color::make_rgba(r, g, b);
                                        if (    ncv::color::make_red(rgba) != r ||
                                                ncv::color::make_green(rgba) != g ||
                                                ncv::color::make_blue(rgba) != b)
                                        {
                                                ncv::log_error() << "failed!";
                                                return EXIT_FAILURE;
                                        }
                                }
                        }
                }

                ncv::log_info() << ">>> done.";
        }

        // test CIELab transform
        {
                ncv::log_info() << "testing CIELab transform ...";

                ncv::scalar_t min_cie_l = +1e100, min_cie_a = min_cie_l, min_cie_b = min_cie_l;
                ncv::scalar_t max_cie_l = -min_cie_l, max_cie_a = max_cie_l, max_cie_b = max_cie_l;

                for (ncv::rgba_t r = 0; r < 256; r ++)
                {
                        for (ncv::rgba_t g = 0; g < 256; g++)
                        {
                                for (ncv::rgba_t b = 0; b < 256; b ++)
                                {
                                        const ncv::rgba_t rgba = ncv::color::make_rgba(r, g, b);
                                        const ncv::cielab_t cielab = ncv::color::make_cielab(rgba);

                                        if (ncv::color::make_rgba(cielab) != rgba)
                                        {
                                                ncv::log_error() << "failed!";
                                                return EXIT_FAILURE;
                                        }

                                        min_cie_l = std::min(min_cie_l, cielab(0));
                                        min_cie_a = std::min(min_cie_a, cielab(1));
                                        min_cie_b = std::min(min_cie_b, cielab(2));

                                        max_cie_l = std::max(max_cie_l, cielab(0));
                                        max_cie_a = std::max(max_cie_a, cielab(1));
                                        max_cie_b = std::max(max_cie_b, cielab(2));
                                }
                        }
                }

                ncv::log_info() << ">>> done.";

                ncv::log_info() << "CIELab range: L = [" << min_cie_l << ", " << max_cie_l
                                << "], a = [" << min_cie_a << ", " << max_cie_a
                                << "], b = [" << min_cie_b << ", " << max_cie_b << "].";
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
