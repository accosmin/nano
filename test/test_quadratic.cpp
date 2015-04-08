#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_quadratic"

#include <boost/test/unit_test.hpp>
#include "nanocv/random.hpp"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/epsilon.hpp"
#include "nanocv/optim/quadratic.hpp"

BOOST_AUTO_TEST_CASE(test_quadratic)
{
        using namespace ncv;

        const size_t tests = 1024;

        for (size_t t = 0; t < tests; t ++)
        {
                random_t<double> rnd(-1.0, +1.0);

                // build random quadratic
                const double a = rnd();
                const double b = rnd();
                const double c = rnd();

                const double x0 = rnd();
                const double f0 = optim::quadratic_value(a, b, c, x0);
                const double g0 = optim::quadratic_value(a, b, c, x0);

                const double x1 = x0 + rnd() + 1.0;
                const double f1 = optim::quadratic_value(a, b, c, x1);

                // interpolate
                double ia, ib, ic;
                optim::quadratic(x0, f0, g0, x1, f1, ia, ib, ic);

                BOOST_CHECK_LE(math::abs(a - ia), math::epsilon1<double>());
                BOOST_CHECK_LE(math::abs(b - ib), math::epsilon1<double>());
                BOOST_CHECK_LE(math::abs(c - ic), math::epsilon1<double>());
        }
}
