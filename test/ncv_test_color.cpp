#include "nanocv.h"

int main(int argc, char *argv[])
{
        using namespace ncv;

        // test RGBA transform
        {
                log_info() << "testing RGBA transform ...";

                for (rgba_t r = 0; r < 256; r ++)
                {
                        for (rgba_t g = 0; g < 256; g++)
                        {
                                for (rgba_t b = 0; b < 256; b ++)
                                {
                                        for (rgba_t a = 0; a < 256; a ++)
                                        {
                                                const rgba_t rgba = color::make_rgba(r, g, b, a);

                                                if (    color::get_red(rgba)    != r ||
                                                        color::get_green(rgba)  != g ||
                                                        color::get_blue(rgba)   != b ||
                                                        color::get_alpha(rgba)  != a)
                                                {
                                                        log_error() << "RGBA setup failed!";
                                                        return EXIT_FAILURE;
                                                }

                                                if (    rgba != color::set_red(rgba, r) ||
                                                        rgba != color::set_green(rgba, g) ||
                                                        rgba != color::set_blue(rgba, b) ||
                                                        rgba != color::set_alpha(rgba, a))
                                                {
                                                        log_error() << "RGBA update failed!";
                                                        return EXIT_FAILURE;
                                                }
                                        }
                                }
                        }
                }

                log_info() << ">>> done.";
        }

        // test CIELab transform
        {
                log_info() << "testing CIELab transform ...";

                scalar_t min_cie_l = +1e100, min_cie_a = min_cie_l, min_cie_b = min_cie_l;
                scalar_t max_cie_l = -min_cie_l, max_cie_a = max_cie_l, max_cie_b = max_cie_l;

                for (rgba_t r = 0; r < 256; r ++)
                {
                        for (rgba_t g = 0; g < 256; g++)
                        {
                                for (rgba_t b = 0; b < 256; b ++)
                                {
                                        const rgba_t rgba = color::make_rgba(r, g, b);
                                        const cielab_t cielab = color::make_cielab(rgba);

                                        if (color::make_rgba(cielab) != rgba)
                                        {
                                                log_error() << "failed!";
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

                log_info() << ">>> done.";

                log_info() << "CIELab range: L = [" << min_cie_l << ", " << max_cie_l
                                << "], a = [" << min_cie_a << ", " << max_cie_a
                                << "], b = [" << min_cie_b << ", " << max_cie_b << "].";
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
