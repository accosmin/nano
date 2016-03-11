#include "unit_test.hpp"
#include "vision/color.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"

NANO_BEGIN_MODULE(test_color)

NANO_CASE(rgba_transform)
{
        using namespace nano;

        const size_t tests = 1023;

        nano::random_t<rgba_t> rgen(0, 255);
        nano::random_t<rgba_t> ggen(0, 255);
        nano::random_t<rgba_t> bgen(0, 255);
        nano::random_t<rgba_t> agen(0, 255);

        for (size_t t = 0; t < tests; ++ t)
        {
                const rgba_t r = rgen();
                const rgba_t g = ggen();
                const rgba_t b = bgen();
                const rgba_t a = agen();

                const rgba_t rgba = color::make_rgba(r, g, b, a);

                NANO_CHECK_EQUAL(color::make_luma(rgba), color::make_luma(r, g, b));

                NANO_CHECK_EQUAL(r, color::get_red(rgba));
                NANO_CHECK_EQUAL(g, color::get_green(rgba));
                NANO_CHECK_EQUAL(b, color::get_blue(rgba));
                NANO_CHECK_EQUAL(a, color::get_alpha(rgba));

                NANO_CHECK_EQUAL(rgba, color::set_red(rgba, r));
                NANO_CHECK_EQUAL(rgba, color::set_green(rgba, g));
                NANO_CHECK_EQUAL(rgba, color::set_blue(rgba, b));
                NANO_CHECK_EQUAL(rgba, color::set_alpha(rgba, a));

                NANO_CHECK_EQUAL(r, color::make_luma(color::make_rgba(r, r, r)));
                NANO_CHECK_EQUAL(g, color::make_luma(color::make_rgba(g, g, g)));
                NANO_CHECK_EQUAL(b, color::make_luma(color::make_rgba(b, b, b)));
        }
}

NANO_CASE(color_tensor)
{
        using namespace nano;

        const int tests = 17;

        for (int test = 0; test < tests; ++ test)
        {
                nano::random_t<int> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();

                const auto eps = nano::epsilon0<scalar_t>();

                // load from RGBA tensor
                {
                        tensor_t data(4, rows, cols);
                        data.matrix(0).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(1).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(2).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(3).setConstant(((test * rng()) % 256) / 255.0);

                        const auto rgba = color::from_rgba_tensor(data);
                        NANO_CHECK_EQUAL(rgba.rows(), data.rows());
                        NANO_CHECK_EQUAL(rgba.cols(), data.cols());

                        const auto idata = color::to_rgba_tensor(rgba);

                        NANO_REQUIRE_EQUAL(data.dims(), idata.dims());
                        NANO_REQUIRE_EQUAL(data.rows(), idata.rows());
                        NANO_REQUIRE_EQUAL(data.cols(), idata.cols());
                        NANO_CHECK_EIGEN_CLOSE(data.vector(), idata.vector(), eps);
                }

                // load from RGB tensor
                {
                        tensor_t data(3, rows, cols);
                        data.matrix(0).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(1).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(2).setConstant(((test * rng()) % 256) / 255.0);

                        const auto rgba = color::from_rgb_tensor(data);
                        NANO_CHECK_EQUAL(rgba.rows(), data.rows());
                        NANO_CHECK_EQUAL(rgba.cols(), data.cols());

                        const auto idata = color::to_rgb_tensor(rgba);

                        NANO_REQUIRE_EQUAL(data.dims(), idata.dims());
                        NANO_REQUIRE_EQUAL(data.rows(), idata.rows());
                        NANO_REQUIRE_EQUAL(data.cols(), idata.cols());
                        NANO_CHECK_EIGEN_CLOSE(data.vector(), idata.vector(), eps);
                }

                // load from LUMA tensor
                {
                        tensor_t data(1, rows, cols);
                        data.matrix(0).setConstant(((test * rng()) % 256) / 255.0);

                        const auto luma = color::from_luma_tensor(data);
                        NANO_CHECK_EQUAL(luma.rows(), data.rows());
                        NANO_CHECK_EQUAL(luma.cols(), data.cols());

                        const auto idata = color::to_luma_tensor(luma);

                        NANO_REQUIRE_EQUAL(data.dims(), idata.dims());
                        NANO_REQUIRE_EQUAL(data.rows(), idata.rows());
                        NANO_REQUIRE_EQUAL(data.cols(), idata.cols());
                        NANO_CHECK_EIGEN_CLOSE(data.vector(), idata.vector(), eps);
                }
        }
}

NANO_END_MODULE()
