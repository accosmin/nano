#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_image"

#include <boost/test/unit_test.hpp>
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "cortex/vision/image.h"
#include <cstdio>

BOOST_AUTO_TEST_CASE(test_image_construct)
{
        using namespace ncv;

        for (size_t test = 0; test < 16; test ++)
        {
                math::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();
                const auto mode = (test % 2 == 0) ? color_mode::luma : color_mode::rgba;

                image_t image(rows, cols, mode);

                BOOST_CHECK_EQUAL(image.is_luma(), mode == color_mode::luma);
                BOOST_CHECK_EQUAL(image.is_rgba(), mode == color_mode::rgba);
                BOOST_CHECK_EQUAL(image.rows(), rows);
                BOOST_CHECK_EQUAL(image.cols(), cols);
                BOOST_CHECK_EQUAL(image.size(), rows * cols);
                BOOST_CHECK(image.mode() == mode);

                BOOST_CHECK_EQUAL(image.luma().rows(), image.is_luma() ? rows : 0);
                BOOST_CHECK_EQUAL(image.luma().cols(), image.is_luma() ? cols : 0);

                BOOST_CHECK_EQUAL(image.rgba().rows(), image.is_rgba() ? rows : 0);
                BOOST_CHECK_EQUAL(image.rgba().cols(), image.is_rgba() ? cols : 0);

                image.make_luma();
                BOOST_CHECK_EQUAL(image.is_luma(), true);
                BOOST_CHECK(image.mode() == color_mode::luma);

                image.make_rgba();
                BOOST_CHECK_EQUAL(image.is_rgba(), true);
                BOOST_CHECK(image.mode() == color_mode::rgba);
        }
}

BOOST_AUTO_TEST_CASE(test_image_io_matrix)
{
        using namespace ncv;

        for (size_t test = 0; test < 16; test ++)
        {
                math::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();

                {
                        // load RGBA as RGBA
                        rgba_matrix_t data(rows, cols);
                        data.setConstant(color::make_random_rgba());

                        image_t image;
                        BOOST_CHECK_EQUAL(image.load_rgba(data), true);
                        BOOST_CHECK_EQUAL(image.is_luma(), false);
                        BOOST_CHECK_EQUAL(image.is_rgba(), true);
                        BOOST_CHECK_EQUAL(image.rows(), rows);
                        BOOST_CHECK_EQUAL(image.cols(), cols);
                        BOOST_CHECK(image.mode() == color_mode::rgba);

                        const auto& rgba = image.rgba();
                        BOOST_REQUIRE_EQUAL(rgba.size(), data.size());

                        for (int i = 0; i < rgba.size(); i ++)
                        {
                                BOOST_CHECK_EQUAL(rgba(i), data(i));
                        }
                }

                {
                        // load RGBA as LUMA
                        rgba_matrix_t data(rows, cols);
                        data.setConstant(color::make_random_rgba());

                        image_t image;
                        BOOST_CHECK_EQUAL(image.load_luma(data), true);
                        BOOST_CHECK_EQUAL(image.is_luma(), true);
                        BOOST_CHECK_EQUAL(image.is_rgba(), false);
                        BOOST_CHECK_EQUAL(image.rows(), rows);
                        BOOST_CHECK_EQUAL(image.cols(), cols);
                        BOOST_CHECK(image.mode() == color_mode::luma);

                        const auto& luma = image.luma();
                        BOOST_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); i ++)
                        {
                                BOOST_CHECK_EQUAL(luma(i), color::make_luma(data(i)));
                        }
                }

                {
                        // load LUMA as LUMA
                        luma_matrix_t data(rows, cols);
                        data.setConstant(color::make_luma(color::make_random_rgba()));

                        image_t image;
                        BOOST_CHECK_EQUAL(image.load_luma(data), true);
                        BOOST_CHECK_EQUAL(image.is_luma(), true);
                        BOOST_CHECK_EQUAL(image.is_rgba(), false);
                        BOOST_CHECK_EQUAL(image.rows(), rows);
                        BOOST_CHECK_EQUAL(image.cols(), cols);
                        BOOST_CHECK(image.mode() == color_mode::luma);

                        const auto& luma = image.luma();
                        BOOST_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); i ++)
                        {
                                BOOST_CHECK_EQUAL(luma(i), data(i));
                        }
                }
        }
}

BOOST_AUTO_TEST_CASE(test_image_io_file)
{
        using namespace ncv;

        for (size_t test = 0; test < 16; test ++)
        {
                math::random_t<coord_t> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();

                rgba_matrix_t data(rows, cols);
                data.setConstant(color::make_random_rgba());

                image_t image;
                BOOST_CHECK_EQUAL(image.load_rgba(data), true);

                const auto path = "test-image.png";

                BOOST_CHECK_EQUAL(image.save(path), true);

                // load RGBA as LUMA
                {
                        BOOST_CHECK_EQUAL(image.load_luma(path), true);
                        BOOST_CHECK_EQUAL(image.is_luma(), true);
                        BOOST_CHECK_EQUAL(image.is_rgba(), false);
                        BOOST_CHECK_EQUAL(image.rows(), rows);
                        BOOST_CHECK_EQUAL(image.cols(), cols);
                        BOOST_CHECK(image.mode() == color_mode::luma);

                        const auto& luma = image.luma();
                        BOOST_REQUIRE_EQUAL(luma.size(), data.size());

                        for (int i = 0; i < luma.size(); i ++)
                        {
                                BOOST_CHECK_EQUAL(luma(i), color::make_luma(data(i)));
                        }
                }

                // load RGBA as RGBA
                {
                        BOOST_CHECK_EQUAL(image.load_rgba(path), true);
                        BOOST_CHECK_EQUAL(image.is_luma(), false);
                        BOOST_CHECK_EQUAL(image.is_rgba(), true);
                        BOOST_CHECK_EQUAL(image.rows(), rows);
                        BOOST_CHECK_EQUAL(image.cols(), cols);
                        BOOST_CHECK(image.mode() == color_mode::rgba);

                        const auto& rgba = image.rgba();
                        BOOST_REQUIRE_EQUAL(rgba.size(), data.size());

                        for (int i = 0; i < rgba.size(); i ++)
                        {
                                BOOST_CHECK_EQUAL(rgba(i), data(i));
                        }
                }

                // cleanup
                std::remove(path);
        }
}

