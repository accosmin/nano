#include "unit_test.hpp"
#include "vision/image.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include <cstdio>

NANO_BEGIN_MODULE(test_image)

NANO_CASE(construct)
{
        using namespace nano;

        for (size_t test = 0; test < 16; ++ test)
        {
                nano::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();
                const auto mode = (test % 2 == 0) ? color_mode::luma : color_mode::rgba;

                image_t image(rows, cols, mode);

                NANO_CHECK_EQUAL(image.is_luma(), mode == color_mode::luma);
                NANO_CHECK_EQUAL(image.is_rgba(), mode == color_mode::rgba);
                NANO_CHECK_EQUAL(image.rows(), rows);
                NANO_CHECK_EQUAL(image.cols(), cols);
                NANO_CHECK_EQUAL(image.size(), rows * cols);
                NANO_CHECK_EQUAL(image.mode(), mode);

                NANO_CHECK_EQUAL(image.luma().rows(), image.is_luma() ? rows : 0);
                NANO_CHECK_EQUAL(image.luma().cols(), image.is_luma() ? cols : 0);

                NANO_CHECK_EQUAL(image.rgba().rows(), image.is_rgba() ? rows : 0);
                NANO_CHECK_EQUAL(image.rgba().cols(), image.is_rgba() ? cols : 0);

                image.make_luma();
                NANO_CHECK_EQUAL(image.is_luma(), true);
                NANO_CHECK_EQUAL(image.mode(), color_mode::luma);

                image.make_rgba();
                NANO_CHECK_EQUAL(image.is_rgba(), true);
                NANO_CHECK_EQUAL(image.mode(), color_mode::rgba);
        }
}

NANO_CASE(io_matrix)
{
        using namespace nano;

        for (size_t test = 0; test < 16; ++ test)
        {
                nano::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();

                {
                        // load RGBA as RGBA
                        rgba_matrix_t data(rows, cols);
                        data.setConstant(color::make_random_rgba());

                        image_t image;
                        NANO_CHECK_EQUAL(image.load_rgba(data), true);
                        NANO_CHECK_EQUAL(image.is_luma(), false);
                        NANO_CHECK_EQUAL(image.is_rgba(), true);
                        NANO_CHECK_EQUAL(image.rows(), rows);
                        NANO_CHECK_EQUAL(image.cols(), cols);
                        NANO_CHECK_EQUAL(image.mode(), color_mode::rgba);

                        const auto& rgba = image.rgba();
                        NANO_REQUIRE_EQUAL(rgba.size(), data.size());

                        for (int i = 0; i < rgba.size(); ++ i)
                        {
                                NANO_CHECK_EQUAL(rgba(i), data(i));
                        }
                }

                {
                        // load RGBA as LUMA
                        rgba_matrix_t data(rows, cols);
                        data.setConstant(color::make_random_rgba());

                        image_t image;
                        NANO_CHECK_EQUAL(image.load_luma(data), true);
                        NANO_CHECK_EQUAL(image.is_luma(), true);
                        NANO_CHECK_EQUAL(image.is_rgba(), false);
                        NANO_CHECK_EQUAL(image.rows(), rows);
                        NANO_CHECK_EQUAL(image.cols(), cols);
                        NANO_CHECK_EQUAL(image.mode(), color_mode::luma);

                        const auto& luma = image.luma();
                        NANO_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); ++ i)
                        {
                                NANO_CHECK_EQUAL(luma(i), color::make_luma(data(i)));
                        }
                }

                {
                        // load LUMA as LUMA
                        luma_matrix_t data(rows, cols);
                        data.setConstant(color::make_luma(color::make_random_rgba()));

                        image_t image;
                        NANO_CHECK_EQUAL(image.load_luma(data), true);
                        NANO_CHECK_EQUAL(image.is_luma(), true);
                        NANO_CHECK_EQUAL(image.is_rgba(), false);
                        NANO_CHECK_EQUAL(image.rows(), rows);
                        NANO_CHECK_EQUAL(image.cols(), cols);
                        NANO_CHECK_EQUAL(image.mode(), color_mode::luma);

                        const auto& luma = image.luma();
                        NANO_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); ++ i)
                        {
                                NANO_CHECK_EQUAL(luma(i), data(i));
                        }
                }
        }
}

NANO_CASE(io_file)
{
        using namespace nano;

        for (size_t test = 0; test < 16; ++ test)
        {
                nano::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();

                rgba_matrix_t data(rows, cols);
                data.setConstant(color::make_random_rgba());

                image_t image;
                NANO_CHECK_EQUAL(image.load_rgba(data), true);

                const auto path = "test-image.png";

                NANO_CHECK_EQUAL(image.save(path), true);

                // load RGBA as LUMA
                {
                        NANO_CHECK_EQUAL(image.load_luma(path), true);
                        NANO_CHECK_EQUAL(image.is_luma(), true);
                        NANO_CHECK_EQUAL(image.is_rgba(), false);
                        NANO_CHECK_EQUAL(image.rows(), rows);
                        NANO_CHECK_EQUAL(image.cols(), cols);
                        NANO_CHECK_EQUAL(image.mode(), color_mode::luma);

                        const auto& luma = image.luma();
                        NANO_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); ++ i)
                        {
                                NANO_CHECK_EQUAL(luma(i), color::make_luma(data(i)));
                        }
                }

                // load RGBA as RGBA
                {
                        NANO_CHECK_EQUAL(image.load_rgba(path), true);
                        NANO_CHECK_EQUAL(image.is_luma(), false);
                        NANO_CHECK_EQUAL(image.is_rgba(), true);
                        NANO_CHECK_EQUAL(image.rows(), rows);
                        NANO_CHECK_EQUAL(image.cols(), cols);
                        NANO_CHECK_EQUAL(image.mode(), color_mode::rgba);

                        const auto& rgba = image.rgba();
                        NANO_REQUIRE_EQUAL(rgba.size(), data.size());

                        for (int i = 0; i < rgba.size(); ++ i)
                        {
                                NANO_CHECK_EQUAL(rgba(i), data(i));
                        }
                }

                // cleanup
                std::remove(path);
        }
}

NANO_END_MODULE()
