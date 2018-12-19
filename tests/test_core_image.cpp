#include <utest/utest.h>
#include "core/image.h"
#include "core/random.h"
#include <cstdio>

using namespace nano;

UTEST_BEGIN_MODULE(test_core_image)

UTEST_CASE(construction)
{
        auto rng = make_rng();
        auto udist = make_udist<coord_t>(16, 64);

        const auto rows = udist(rng);
        const auto cols = udist(rng);

        image_t image(rows, cols, color_mode::luma);
        UTEST_CHECK_EQUAL(image.is_luma(), true);
        UTEST_CHECK_EQUAL(image.is_rgb(), false);
        UTEST_CHECK_EQUAL(image.is_rgba(), false);
        UTEST_CHECK_EQUAL(image.dims(), 1);
        UTEST_CHECK_EQUAL(image.rows(), rows);
        UTEST_CHECK_EQUAL(image.cols(), cols);
        UTEST_CHECK_EQUAL(image.size(), rows * cols);
        UTEST_CHECK_EQUAL(image.mode(), color_mode::luma);

        image.make_luma();
        UTEST_CHECK_EQUAL(image.dims(), 1);
        UTEST_CHECK_EQUAL(image.mode(), color_mode::luma);

        image.make_rgb();
        UTEST_CHECK_EQUAL(image.dims(), 3);
        UTEST_CHECK_EQUAL(image.mode(), color_mode::rgb);

        image.make_rgba();
        UTEST_CHECK_EQUAL(image.dims(), 4);
        UTEST_CHECK_EQUAL(image.mode(), color_mode::rgba);
}

UTEST_CASE(transform)
{
        auto rng = make_rng();
        auto udist = make_udist<coord_t>(16, 64);

        const auto rows = udist(rng);
        const auto cols = udist(rng);

        const auto color_rgb = rgb_t{54, 3, 217};
        const auto color_rgba = rgba_t{color_rgb(0), color_rgb(1), color_rgb(2), 45};
        const auto color_luma = static_cast<luma_t>(make_luma(color_rgb(0), color_rgb(1), color_rgb(2)));

        image_t image(rows, cols, color_mode::rgba);
        UTEST_CHECK_EQUAL(image.dims(), 4);
        UTEST_CHECK_EQUAL(image.rows(), rows);
        UTEST_CHECK_EQUAL(image.cols(), cols);

        image.fill(color_rgba);
        UTEST_REQUIRE_EQUAL(image.dims(), 4);
        for (auto i = 0; i < image.dims(); ++ i)
        {
                UTEST_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgba(i));
                UTEST_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgba(i));
        }

        image.make_rgb();
        UTEST_REQUIRE_EQUAL(image.dims(), 3);
        for (auto i = 0; i < image.dims(); ++ i)
        {
                UTEST_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgb(i));
                UTEST_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgb(i));
        }

        image.make_luma();
        UTEST_REQUIRE_EQUAL(image.dims(), 1);
        for (auto i = 0; i < image.dims(); ++ i)
        {
                UTEST_CHECK_EQUAL(image.plane(i).minCoeff(), color_luma);
                UTEST_CHECK_EQUAL(image.plane(i).maxCoeff(), color_luma);
        }
}

UTEST_CASE(io_tensor)
{
        auto rng = make_rng();
        auto udist = make_udist<coord_t>(16, 64);

        const auto rows = udist(rng);
        const auto cols = udist(rng);

        const auto color_rgb = rgb_t{54, 3, 217};
        const auto color_rgba = rgba_t{color_rgb(0), color_rgb(1), color_rgb(2), 45};

        image_t image(rows, cols, color_mode::rgb);
        image.fill(color_rgba);

        const auto tensor = image.to_tensor();
        UTEST_CHECK_EQUAL(tensor.size<0>(), image.dims());
        UTEST_CHECK_EQUAL(tensor.size<1>(), image.rows());
        UTEST_CHECK_EQUAL(tensor.size<2>(), image.cols());

        UTEST_REQUIRE(image.from_tensor(tensor));
        UTEST_CHECK_EQUAL(tensor.size<0>(), image.dims());
        UTEST_CHECK_EQUAL(tensor.size<1>(), image.rows());
        UTEST_CHECK_EQUAL(tensor.size<2>(), image.cols());

        UTEST_REQUIRE_EQUAL(image.dims(), 3);
        for (auto i = 0; i < image.dims(); ++ i)
        {
                UTEST_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgb(i));
                UTEST_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgb(i));
        }
}

UTEST_CASE(io_luma)
{
        auto rng = make_rng();
        auto udist = make_udist<coord_t>(16, 64);

        const auto rows = udist(rng);
        const auto cols = udist(rng);

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

                UTEST_CHECK_EQUAL(image.save(path), true);
        }

        {
                image_t image;
                UTEST_CHECK_EQUAL(image.load_luma(path), true);
                UTEST_CHECK_EQUAL(image.rows(), rows);
                UTEST_CHECK_EQUAL(image.cols(), cols);
                UTEST_REQUIRE_EQUAL(image.mode(), color_mode::luma);

                UTEST_REQUIRE_EQUAL(image.dims(), 1);
                UTEST_CHECK_EQUAL(image.plane(0, r00).minCoeff(), c00);
                UTEST_CHECK_EQUAL(image.plane(0, r00).maxCoeff(), c00);

                UTEST_CHECK_EQUAL(image.plane(0, r01).minCoeff(), c01);
                UTEST_CHECK_EQUAL(image.plane(0, r01).maxCoeff(), c01);

                UTEST_CHECK_EQUAL(image.plane(0, r10).minCoeff(), c10);
                UTEST_CHECK_EQUAL(image.plane(0, r10).maxCoeff(), c10);

                UTEST_CHECK_EQUAL(image.plane(0, r11).minCoeff(), c11);
                UTEST_CHECK_EQUAL(image.plane(0, r11).maxCoeff(), c11);
        }

        // cleanup
        std::remove(path);
}

UTEST_CASE(io_file)
{
        auto rng = make_rng();
        auto udist = make_udist<coord_t>(16, 64);

        const auto rows = udist(rng);
        const auto cols = udist(rng);

        const auto color_rgb = rgb_t{54, 3, 217};
        const auto color_rgba = rgba_t{color_rgb(0), color_rgb(1), color_rgb(2), 45};
        const auto color_luma = static_cast<luma_t>(make_luma(color_rgb(0), color_rgb(1), color_rgb(2)));

        image_t orig_image(rows, cols, color_mode::rgba);
        orig_image.fill(color_rgba);

        const auto path = "test-image.png";

        UTEST_CHECK_EQUAL(orig_image.save(path), true);

        // load as LUMA
        {
                image_t image;
                UTEST_CHECK_EQUAL(image.load_luma(path), true);
                UTEST_CHECK_EQUAL(image.rows(), rows);
                UTEST_CHECK_EQUAL(image.cols(), cols);
                UTEST_REQUIRE_EQUAL(image.mode(), color_mode::luma);

                UTEST_REQUIRE_EQUAL(image.dims(), 1);
                for (auto i = 0; i < image.dims(); ++ i)
                {
                        UTEST_CHECK_EQUAL(image.plane(i).minCoeff(), color_luma);
                        UTEST_CHECK_EQUAL(image.plane(i).maxCoeff(), color_luma);
                }
        }

        // load as RGBA
        {
                image_t image;
                UTEST_CHECK_EQUAL(image.load_rgba(path), true);
                UTEST_CHECK_EQUAL(image.rows(), rows);
                UTEST_CHECK_EQUAL(image.cols(), cols);
                UTEST_CHECK_EQUAL(image.mode(), color_mode::rgba);

                UTEST_REQUIRE_EQUAL(image.dims(), 4);
                for (auto i = 0; i < image.dims(); ++ i)
                {
                        UTEST_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgba(i));
                        UTEST_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgba(i));
                }
        }

        // load as RGB
        {
                image_t image;
                UTEST_CHECK_EQUAL(image.load_rgb(path), true);
                UTEST_CHECK_EQUAL(image.rows(), rows);
                UTEST_CHECK_EQUAL(image.cols(), cols);
                UTEST_CHECK_EQUAL(image.mode(), color_mode::rgb);

                UTEST_REQUIRE_EQUAL(image.dims(), 3);
                for (auto i = 0; i < image.dims(); ++ i)
                {
                        UTEST_CHECK_EQUAL(image.plane(i).minCoeff(), color_rgb(i));
                        UTEST_CHECK_EQUAL(image.plane(i).maxCoeff(), color_rgb(i));
                }
        }

        // cleanup
        std::remove(path);
}

UTEST_END_MODULE()
