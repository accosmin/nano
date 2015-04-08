#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_quadratic"

#include <boost/test/unit_test.hpp>
#include "nanocv/random.hpp"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/epsilon.hpp"
#include "nanocv/optim/quadratic.hpp"

#include <iostream>

BOOST_AUTO_TEST_CASE(test_quadratic)
{
        using namespace ncv;

        const size_t tests = 1;//1024;

        for (size_t t = 0; t < tests; t ++)
        {
                random_t<double> rnd(-1.0, +1.0);

                // build random quadratic
                const double a = rnd() + 1.0; // convex!
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

                BOOST_CHECK_LE(math::abs(f0 - optim::quadratic_value(ia, ib, ic, x0)), math::epsilon0<double>());
                BOOST_CHECK_LE(math::abs(g0 - optim::quadratic_grad(ia, ib, ic, x0)), math::epsilon0<double>());

                BOOST_CHECK_LE(math::abs(f1 - optim::quadratic_value(ia, ib, ic, x1)), math::epsilon0<double>());

                BOOST_CHECK_LE(math::abs(a - ia), math::epsilon1<double>());
                BOOST_CHECK_LE(math::abs(b - ib), math::epsilon1<double>());
                BOOST_CHECK_LE(math::abs(c - ic), math::epsilon1<double>());

                std::cout << "(" << a << ", " << b << ", " << c
                          << ") vs (" << ia << ", " << ib << ", " << ic << ")\n";

                std::cout << "x0 = " << x0 << ", x1 = " << x1 << "\n";

                // extremum
                double min;
                optim::quadratic_extremum(a, b, c, min);

                const size_t etests = 4;//28;
                for (size_t e = 0; e < etests; e ++)
                {
                        BOOST_CHECK_LE(optim::quadratic_value(ia, ib, ic, min),
                                       optim::quadratic_value(ia, ib, ic, rnd()));
                }
        }
}
