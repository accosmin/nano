#include "utest.h"
#include "core/random.h"
#include "core/quadratic.h"

NANO_BEGIN_MODULE(test_quadratic)

NANO_CASE(evaluate)
{
        for (auto t = 0; t < 127; ++ t)
        {
                auto rng = nano::make_rng();
                auto rnd = nano::make_udist<double>(-1.0, +1.0);

                // build random valid quadratic
                double a, b, c;
                do
                {
                        a = rnd(rng);
                        b = rnd(rng);
                        c = rnd(rng);
                }
                while (!nano::quadratic_t<double>(a, b, c));

                const nano::quadratic_t<double> q(a, b, c);
                NANO_CHECK(q);

                const double x0 = rnd(rng);
                const double f0 = q.value(x0);
                const double g0 = q.gradient(x0);

                const double x1 = x0 + rnd(rng) + 1.0;
                const double f1 = q.value(x1);

                // check interpolation
                const nano::quadratic_t<double> iq(x0, f0, g0, x1, f1);
                if (!iq)
                {
                        continue;
                }

                NANO_CHECK_CLOSE(f0, iq.value(x0), nano::epsilon0<double>());
                NANO_CHECK_CLOSE(g0, iq.gradient(x0), nano::epsilon0<double>());

                NANO_CHECK_CLOSE(f1, iq.value(x1), nano::epsilon0<double>());

//                NANO_CHECK_CLOSE(q.a(), iq.a(), nano::epsilon1<double>());
//                NANO_CHECK_CLOSE(q.b(), iq.b(), nano::epsilon1<double>());
//                NANO_CHECK_CLOSE(q.c(), iq.c(), nano::epsilon1<double>());

                // check extremum
                double extremum;
                iq.extremum(extremum);

                if (!std::isfinite(extremum))
                {
                        continue;
                }

                NANO_CHECK_LESS(nano::abs(iq.gradient(extremum)), nano::epsilon0<double>());

                for (auto e = 0; e < 143; ++ e)
                {
                        NANO_CHECK_GREATER(nano::abs(iq.gradient(rnd(rng))), nano::abs(iq.gradient(extremum)));
                }
        }
}

NANO_END_MODULE()
