#include "unit_test.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "math/quadratic.hpp"

ZOB_BEGIN_MODULE(test_quadratic)

ZOB_CASE(evaluate)
{
        const size_t tests = 127;

        for (size_t t = 0; t < tests; ++ t)
        {
                auto rnd = zob::make_rng<double>(-1.0, +1.0);

                // build random valid quadratic
                double a, b, c;
                do
                {
                        a = rnd();
                        b = rnd();
                        c = rnd();
                }
                while (!zob::quadratic_t<double>(a, b, c));

                const zob::quadratic_t<double> q(a, b, c);
                ZOB_CHECK(q);

                const double x0 = rnd();
                const double f0 = q.value(x0);
                const double g0 = q.gradient(x0);

                const double x1 = x0 + rnd() + 1.0;
                const double f1 = q.value(x1);

                // check interpolation
                const zob::quadratic_t<double> iq(x0, f0, g0, x1, f1);
                if (!iq)
                {
                        continue;
                }

                ZOB_CHECK_CLOSE(f0, iq.value(x0), zob::epsilon0<double>());
                ZOB_CHECK_CLOSE(g0, iq.gradient(x0), zob::epsilon0<double>());

                ZOB_CHECK_CLOSE(f1, iq.value(x1), zob::epsilon0<double>());

//                ZOB_CHECK_CLOSE(q.a(), iq.a(), zob::epsilon1<double>());
//                ZOB_CHECK_CLOSE(q.b(), iq.b(), zob::epsilon1<double>());
//                ZOB_CHECK_CLOSE(q.c(), iq.c(), zob::epsilon1<double>());

                // check extremum
                double extremum;
                iq.extremum(extremum);

                if (!std::isfinite(extremum))
                {
                        continue;
                }

                ZOB_CHECK_LESS(zob::abs(iq.gradient(extremum)), zob::epsilon0<double>());

                const size_t etests = 143;
                for (size_t e = 0; e < etests; ++ e)
                {
                        ZOB_CHECK_GREATER(zob::abs(iq.gradient(rnd())), zob::abs(iq.gradient(extremum)));
                }
        }
}

ZOB_END_MODULE()
