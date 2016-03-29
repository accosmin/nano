#include "unit_test.hpp"
#include "vision/image.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include <cstdio>

NANO_BEGIN_MODULE(test_image)

NANO_CASE(construction)
{
        using namespace nano;

        nano::random_t<coord_t> rng(16, 64);

        const auto rows = rng();
        const auto cols = rng();

        image_t image(rows, cols, color_mode::luma);
        NANO_CHECK_EQUAL(image.is_luma(), true);
        NANO_CHECK_EQUAL(image.is_rgb(), false);
        NANO_CHECK_EQUAL(image.is_rgba(), false);
        NANO_CHECK_EQUAL(image.dims(), 1);
        NANO_CHECK_EQUAL(image.rows(), rows);
        NANO_CHECK_EQUAL(image.cols(), cols);
        NANO_CHECK_EQUAL(image.size(), rows * cols);
        NANO_CHECK_EQUAL(image.mode(), color_mode::luma);

        image.make_luma();
        NANO_CHECK_EQUAL(image.dims(), 1);
        NANO_CHECK_EQUAL(image.mode(), color_mode::luma);

        image.make_rgb();
        NANO_CHECK_EQUAL(image.dims(), 3);
        NANO_CHECK_EQUAL(image.mode(), color_mode::rgb);

        image.make_rgba();
        NANO_CHECK_EQUAL(image.dims(), 4);
        NANO_CHECK_EQUAL(image.mode(), color_mode::rgba);
}

NANO_CASE(transform)
{
        using namespace nano;

        nano::random_t<coord_t> rng(16, 64);

        const auto rows = rng();
        const auto cols = rng();

        const auto color_rgb = rgb_t{54, 3, 217};
        const auto color_rgba = rgba_t{color_rgb(0), color_rgb(1), color_rgb(2), 45};
        const auto color_luma = static_cast<luma_t>(make_luma(color_rgb(0), color_rgb(1), color_rgb(2)));

        image_t image(rows, cols, color_mode::rgba);
        NANO_CHECK_EQUAL(image.dims(), 4);
        NANO_CHECK_EQUAL(image.rows(), rows);
        NANO_CHECK_EQUAL(image.cols(), cols);

        image.fill(color_rgba);
        NANO_REQUIRE_EQUAL(image.dims(), 4);
        for (auto i = 0; i < image.dims(); ++ i)
        {
                NANO_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgba(i));
                NANO_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgba(i));
        }

        image.make_rgb();
        NANO_REQUIRE_EQUAL(image.dims(), 3);
        for (auto i = 0; i < image.dims(); ++ i)
        {
                NANO_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgb(i));
                NANO_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgb(i));
        }

        image.make_luma();
        NANO_REQUIRE_EQUAL(image.dims(), 1);
        for (auto i = 0; i < image.dims(); ++ i)
        {
                NANO_CHECK_EQUAL(image.plane(i).minCoeff(), color_luma);
                NANO_CHECK_EQUAL(image.plane(i).maxCoeff(), color_luma);
        }
}

NANO_CASE(io_tensor)
{
        using namespace nano;

        nano::random_t<coord_t> rng(16, 64);

        const auto rows = rng();
        const auto cols = rng();

        const auto color_rgb = rgb_t{54, 3, 217};
        const auto color_rgba = rgba_t{color_rgb(0), color_rgb(1), color_rgb(2), 45};

        image_t image(rows, cols, color_mode::rgb);
        image.fill(color_rgba);

        const auto tensor = image.to_tensor();
        NANO_CHECK_EQUAL(tensor.size<0>(), image.dims());
        NANO_CHECK_EQUAL(tensor.size<1>(), image.rows());
        NANO_CHECK_EQUAL(tensor.size<2>(), image.cols());

        NANO_REQUIRE(image.from_tensor(tensor));
        NANO_CHECK_EQUAL(tensor.size<0>(), image.dims());
        NANO_CHECK_EQUAL(tensor.size<1>(), image.rows());
        NANO_CHECK_EQUAL(tensor.size<2>(), image.cols());

        NANO_REQUIRE_EQUAL(image.dims(), 3);
        for (auto i = 0; i < image.dims(); ++ i)
        {
                NANO_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgb(i));
                NANO_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgb(i));
        }
}

NANO_CASE(io_rgba)
{
        using namespace nano;

        nano::random_t<coord_t> rng(16, 64);

        const auto rows = rng();
        const auto cols = rng();

        const auto dcols = cols / 2;
        const auto drows = rows / 4;

        const auto r00 = rect_t{0, 0, dcols, drows};
        const auto r01 = rect_t{dcols, 0, cols - dcols, drows};
        const auto r10 = rect_t{0, drows, dcols, rows - drows};
        const auto r11 = rect_t{dcols, drows, cols - dcols, rows - drows};

        const auto c00 = luma_t(100);
        const auto c01 = luma_t(101);
        const auto c10 = luma_t(110);
        const auto c11 = luma_t(111);

        const auto path = "test-image.png";

        {
                image_t image(rows, cols, color_mode::luma);
                image.plane(0, r00).setConstant(c00);
                image.plane(0, r01).setConstant(c01);
                image.plane(0, r10).setConstant(c10);
                image.plane(0, r11).setConstant(c11);

                NANO_CHECK_EQUAL(image.save(path), true);
        }

        {
                image_t image;
                NANO_CHECK_EQUAL(image.load_luma(path), true);
                NANO_CHECK_EQUAL(image.rows(), rows);
                NANO_CHECK_EQUAL(image.cols(), cols);
                NANO_REQUIRE_EQUAL(image.mode(), color_mode::luma);

                NANO_REQUIRE_EQUAL(image.dims(), 1);
                NANO_CHECK_EQUAL(image.plane(0, r00).minCoeff(), c00);
                NANO_CHECK_EQUAL(image.plane(0, r00).maxCoeff(), c00);

                NANO_CHECK_EQUAL(image.plane(0, r01).minCoeff(), c01);
                NANO_CHECK_EQUAL(image.plane(0, r01).maxCoeff(), c01);

                NANO_CHECK_EQUAL(image.plane(0, r10).minCoeff(), c10);
                NANO_CHECK_EQUAL(image.plane(0, r10).maxCoeff(), c10);

                NANO_CHECK_EQUAL(image.plane(0, r11).minCoeff(), c11);
                NANO_CHECK_EQUAL(image.plane(0, r11).maxCoeff(), c11);
        }

        // cleanup
        std::remove(path);
}

NANO_CASE(io_file)
{
        using namespace nano;

        nano::random_t<coord_t> rng(16, 64);

        const auto rows = rng();
        const auto cols = rng();

        const auto color_rgb = rgb_t{54, 3, 217};
        const auto color_rgba = rgba_t{color_rgb(0), color_rgb(1), color_rgb(2), 45};
        const auto color_luma = static_cast<luma_t>(make_luma(color_rgb(0), color_rgb(1), color_rgb(2)));

        image_t orig_image(rows, cols, color_mode::rgba);
        orig_image.fill(color_rgba);

        const auto path = "test-image.png";

        NANO_CHECK_EQUAL(orig_image.save(path), true);

        // load as LUMA
        {
                image_t image;
                NANO_CHECK_EQUAL(image.load_luma(path), true);
                NANO_CHECK_EQUAL(image.rows(), rows);
                NANO_CHECK_EQUAL(image.cols(), cols);
                NANO_REQUIRE_EQUAL(image.mode(), color_mode::luma);

                NANO_REQUIRE_EQUAL(image.dims(), 1);
                for (auto i = 0; i < image.dims(); ++ i)
                {
                        NANO_CHECK_EQUAL(image.plane(i).minCoeff(), color_luma);
                        NANO_CHECK_EQUAL(image.plane(i).maxCoeff(), color_luma);
                }
        }

        // load as RGBA
        {
                image_t image;
                NANO_CHECK_EQUAL(image.load_rgba(path), true);
                NANO_CHECK_EQUAL(image.rows(), rows);
                NANO_CHECK_EQUAL(image.cols(), cols);
                NANO_CHECK_EQUAL(image.mode(), color_mode::rgba);

                NANO_REQUIRE_EQUAL(image.dims(), 4);
                for (auto i = 0; i < image.dims(); ++ i)
                {
                        NANO_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgba(i));
                        NANO_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgba(i));
                }
        }

        // load as RGB
        {
                image_t image;
                NANO_CHECK_EQUAL(image.load_rgb(path), true);
                NANO_CHECK_EQUAL(image.rows(), rows);
                NANO_CHECK_EQUAL(image.cols(), cols);
                NANO_CHECK_EQUAL(image.mode(), color_mode::rgb);

                NANO_REQUIRE_EQUAL(image.dims(), 3);
                for (auto i = 0; i < image.dims(); ++ i)
                {
                        NANO_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgb(i));
                        NANO_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgb(i));
                }
        }

        // cleanup
        std::remove(path);
}

NANO_END_MODULE()
