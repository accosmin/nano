#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_color_rgba"

#include <boost/test/unit_test.hpp>
#include "core/color.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"

BOOST_AUTO_TEST_CASE(test_color_rgba_transform)
{
        using namespace ncv;

        const size_t tests = 256 * 1024;

        ncv::random_t<rgba_t> rgen(0, 255);
        ncv::random_t<rgba_t> ggen(0, 255);
        ncv::random_t<rgba_t> bgen(0, 255);
        ncv::random_t<rgba_t> agen(0, 255);

        // test RGBA transform
        for (size_t t = 0; t < tests; t ++)
        {
                const rgba_t r = rgen();
                const rgba_t g = ggen();
                const rgba_t b = bgen();
                const rgba_t a = agen();

                const rgba_t rgba = color::make_rgba(r, g, b, a);

                BOOST_CHECK_EQUAL(color::make_luma(rgba), color::make_luma(r, g, b));

                BOOST_CHECK_EQUAL(r, color::get_red(rgba));
                BOOST_CHECK_EQUAL(g, color::get_green(rgba));
                BOOST_CHECK_EQUAL(b, color::get_blue(rgba));
                BOOST_CHECK_EQUAL(a, color::get_alpha(rgba));

                BOOST_CHECK_EQUAL(rgba, color::set_red(rgba, r));
                BOOST_CHECK_EQUAL(rgba, color::set_green(rgba, g));
                BOOST_CHECK_EQUAL(rgba, color::set_blue(rgba, b));
                BOOST_CHECK_EQUAL(rgba, color::set_alpha(rgba, a));

                BOOST_CHECK_EQUAL(r, color::make_luma(color::make_rgba(r, r, r)));
                BOOST_CHECK_EQUAL(g, color::make_luma(color::make_rgba(g, g, g)));
                BOOST_CHECK_EQUAL(b, color::make_luma(color::make_rgba(b, b, b)));
        }
}

BOOST_AUTO_TEST_CASE(test_color_tensor)
{
        using namespace ncv;

        const size_t tests = 17;

        for (size_t test = 0; test < tests; test ++)
        {
                random_t<int> rng(16, 64);

                const auto rows = rng();
                const auto cols = rng();

                const auto eps = math::epsilon0<scalar_t>();

                // load from RGBA tensor
                {
                        tensor_t data(4, rows, cols);
                        data.matrix(0).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(1).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(2).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(3).setConstant(((test * rng()) % 256) / 255.0);

                        const auto rgba = color::from_rgba_tensor(data);
                        BOOST_CHECK_EQUAL(rgba.rows(), data.rows());
                        BOOST_CHECK_EQUAL(rgba.cols(), data.cols());

                        const auto idata = color::to_rgba_tensor(rgba);

                        BOOST_REQUIRE_EQUAL(data.dims(), idata.dims());
                        BOOST_REQUIRE_EQUAL(data.rows(), idata.rows());
                        BOOST_REQUIRE_EQUAL(data.cols(), idata.cols());
                        BOOST_CHECK_LE((data.vector() - idata.vector()).lpNorm<Eigen::Infinity>(), eps);
                }

                // load from RGB tensor
                {
                        tensor_t data(3, rows, cols);
                        data.matrix(0).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(1).setConstant(((test * rng()) % 256) / 255.0);
                        data.matrix(2).setConstant(((test * rng()) % 256) / 255.0);

                        const auto rgba = color::from_rgb_tensor(data);
                        BOOST_CHECK_EQUAL(rgba.rows(), data.rows());
                        BOOST_CHECK_EQUAL(rgba.cols(), data.cols());

                        const auto idata = color::to_rgb_tensor(rgba);

                        BOOST_REQUIRE_EQUAL(data.dims(), idata.dims());
                        BOOST_REQUIRE_EQUAL(data.rows(), idata.rows());
                        BOOST_REQUIRE_EQUAL(data.cols(), idata.cols());
                        BOOST_CHECK_LE((data.vector() - idata.vector()).lpNorm<Eigen::Infinity>(), eps);
                }

                // load from LUMA tensor
                {
                        tensor_t data(1, rows, cols);
                        data.matrix(0).setConstant(((test * rng()) % 256) / 255.0);

                        const auto luma = color::from_luma_tensor(data);
                        BOOST_CHECK_EQUAL(luma.rows(), data.rows());
                        BOOST_CHECK_EQUAL(luma.cols(), data.cols());

                        const auto idata = color::to_luma_tensor(luma);

                        BOOST_REQUIRE_EQUAL(data.dims(), idata.dims());
                        BOOST_REQUIRE_EQUAL(data.rows(), idata.rows());
                        BOOST_REQUIRE_EQUAL(data.cols(), idata.cols());
                        BOOST_CHECK_LE((data.vector() - idata.vector()).lpNorm<Eigen::Infinity>(), eps);
                }
        }
}
