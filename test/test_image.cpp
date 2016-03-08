#include "unit_test.hpp"
#include "vision/image.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include <cstdio>

ZOB_BEGIN_MODULE(test_image)

ZOB_CASE(construct)
{
        using namespace cortex;

        for (size_t test = 0; test < 16; ++ test)
        {
                math::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();
                const auto mode = (test % 2 == 0) ? color_mode::luma : color_mode::rgba;

                image_t image(rows, cols, mode);

                ZOB_CHECK_EQUAL(image.is_luma(), mode == color_mode::luma);
                ZOB_CHECK_EQUAL(image.is_rgba(), mode == color_mode::rgba);
                ZOB_CHECK_EQUAL(image.rows(), rows);
                ZOB_CHECK_EQUAL(image.cols(), cols);
                ZOB_CHECK_EQUAL(image.size(), rows * cols);
                ZOB_CHECK_EQUAL(image.mode(), mode);

                ZOB_CHECK_EQUAL(image.luma().rows(), image.is_luma() ? rows : 0);
                ZOB_CHECK_EQUAL(image.luma().cols(), image.is_luma() ? cols : 0);

                ZOB_CHECK_EQUAL(image.rgba().rows(), image.is_rgba() ? rows : 0);
                ZOB_CHECK_EQUAL(image.rgba().cols(), image.is_rgba() ? cols : 0);

                image.make_luma();
                ZOB_CHECK_EQUAL(image.is_luma(), true);
                ZOB_CHECK_EQUAL(image.mode(), color_mode::luma);

                image.make_rgba();
                ZOB_CHECK_EQUAL(image.is_rgba(), true);
                ZOB_CHECK_EQUAL(image.mode(), color_mode::rgba);
        }
}

ZOB_CASE(io_matrix)
{
        using namespace cortex;

        for (size_t test = 0; test < 16; ++ test)
        {
                math::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();

                {
                        // load RGBA as RGBA
                        rgba_matrix_t data(rows, cols);
                        data.setConstant(color::make_random_rgba());

                        image_t image;
                        ZOB_CHECK_EQUAL(image.load_rgba(data), true);
                        ZOB_CHECK_EQUAL(image.is_luma(), false);
                        ZOB_CHECK_EQUAL(image.is_rgba(), true);
                        ZOB_CHECK_EQUAL(image.rows(), rows);
                        ZOB_CHECK_EQUAL(image.cols(), cols);
                        ZOB_CHECK_EQUAL(image.mode(), color_mode::rgba);

                        const auto& rgba = image.rgba();
                        ZOB_REQUIRE_EQUAL(rgba.size(), data.size());

                        for (int i = 0; i < rgba.size(); ++ i)
                        {
                                ZOB_CHECK_EQUAL(rgba(i), data(i));
                        }
                }

                {
                        // load RGBA as LUMA
                        rgba_matrix_t data(rows, cols);
                        data.setConstant(color::make_random_rgba());

                        image_t image;
                        ZOB_CHECK_EQUAL(image.load_luma(data), true);
                        ZOB_CHECK_EQUAL(image.is_luma(), true);
                        ZOB_CHECK_EQUAL(image.is_rgba(), false);
                        ZOB_CHECK_EQUAL(image.rows(), rows);
                        ZOB_CHECK_EQUAL(image.cols(), cols);
                        ZOB_CHECK_EQUAL(image.mode(), color_mode::luma);

                        const auto& luma = image.luma();
                        ZOB_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); ++ i)
                        {
                                ZOB_CHECK_EQUAL(luma(i), color::make_luma(data(i)));
                        }
                }

                {
                        // load LUMA as LUMA
                        luma_matrix_t data(rows, cols);
                        data.setConstant(color::make_luma(color::make_random_rgba()));

                        image_t image;
                        ZOB_CHECK_EQUAL(image.load_luma(data), true);
                        ZOB_CHECK_EQUAL(image.is_luma(), true);
                        ZOB_CHECK_EQUAL(image.is_rgba(), false);
                        ZOB_CHECK_EQUAL(image.rows(), rows);
                        ZOB_CHECK_EQUAL(image.cols(), cols);
                        ZOB_CHECK_EQUAL(image.mode(), color_mode::luma);

                        const auto& luma = image.luma();
                        ZOB_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); ++ i)
                        {
                                ZOB_CHECK_EQUAL(luma(i), data(i));
                        }
                }
        }
}

ZOB_CASE(io_file)
{
        using namespace cortex;

        for (size_t test = 0; test < 16; ++ test)
        {
                math::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();

                rgba_matrix_t data(rows, cols);
                data.setConstant(color::make_random_rgba());

                image_t image;
                ZOB_CHECK_EQUAL(image.load_rgba(data), true);

                const auto path = "test-image.png";

                ZOB_CHECK_EQUAL(image.save(path), true);

                // load RGBA as LUMA
                {
                        ZOB_CHECK_EQUAL(image.load_luma(path), true);
                        ZOB_CHECK_EQUAL(image.is_luma(), true);
                        ZOB_CHECK_EQUAL(image.is_rgba(), false);
                        ZOB_CHECK_EQUAL(image.rows(), rows);
                        ZOB_CHECK_EQUAL(image.cols(), cols);
                        ZOB_CHECK_EQUAL(image.mode(), color_mode::luma);

                        const auto& luma = image.luma();
                        ZOB_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); ++ i)
                        {
                                ZOB_CHECK_EQUAL(luma(i), color::make_luma(data(i)));
                        }
                }

                // load RGBA as RGBA
                {
                        ZOB_CHECK_EQUAL(image.load_rgba(path), true);
                        ZOB_CHECK_EQUAL(image.is_luma(), false);
                        ZOB_CHECK_EQUAL(image.is_rgba(), true);
                        ZOB_CHECK_EQUAL(image.rows(), rows);
                        ZOB_CHECK_EQUAL(image.cols(), cols);
                        ZOB_CHECK_EQUAL(image.mode(), color_mode::rgba);

                        const auto& rgba = image.rgba();
                        ZOB_REQUIRE_EQUAL(rgba.size(), data.size());

                        for (int i = 0; i < rgba.size(); ++ i)
                        {
                                ZOB_CHECK_EQUAL(rgba(i), data(i));
                        }
                }

                // cleanup
                std::remove(path);
        }
}

ZOB_END_MODULE()
