#include "utest.h"
#include "math/cubic.h"
#include "math/random.h"
#include "math/numeric.h"
#include "math/epsilon.h"

NANO_BEGIN_MODULE(test_cubic)

NANO_CASE(evaluate)
{
        for (auto t = 0; t < 127; ++ t)
        {
                auto rng = nano::make_rng();
                auto rnd = nano::make_udist<double>(-1.0, +1.0);

                // build random valid cubic
                double a, b, c, d;
                do
                {
                        a = rnd(rng);
                        b = rnd(rng);
                        c = rnd(rng);
                        d = rnd(rng);
                }
                while (!nano::cubic_t<double>(a, b, c, d));

                const nano::cubic_t<double> q(a, b, c, d);
                NANO_CHECK(q);

                const double x0 = rnd(rng);
                const double f0 = q.value(x0);
                const double g0 = q.gradient(x0);

                const double x1 = x0 + rnd(rng) + 1.0;
                const double f1 = q.value(x1);
                const double g1 = q.gradient(x1);

                // check interpolation
                const nano::cubic_t<double> iq(x0, f0, g0, x1, f1, g1);
                if (!iq)
                {
                        continue;
                }

                NANO_CHECK_CLOSE(f0, iq.value(x0), nano::epsilon1<double>());
                NANO_CHECK_CLOSE(g0, iq.gradient(x0), nano::epsilon1<double>());

                NANO_CHECK_CLOSE(f1, iq.value(x1), nano::epsilon1<double>());
                NANO_CHECK_CLOSE(g1, iq.gradient(x1), nano::epsilon1<double>());

//                NANO_CHECK_CLOSE(q.a(), iq.a(), nano::epsilon1<double>());
//                NANO_CHECK_CLOSE(q.b(), iq.b(), nano::epsilon1<double>());
//                NANO_CHECK_CLOSE(q.c(), iq.c(), nano::epsilon1<double>());
//                NANO_CHECK_CLOSE(q.d(), iq.d(), nano::epsilon1<double>());

                // check extremum
                double extremum1, extremum2;
                iq.extremum(extremum1, extremum2);

                if (!std::isfinite(extremum1) || !std::isfinite(extremum2))
                {
                        continue;
                }

                NANO_CHECK_LESS(nano::abs(iq.gradient(extremum1)), nano::epsilon1<double>());
                NANO_CHECK_LESS(nano::abs(iq.gradient(extremum2)), nano::epsilon1<double>());

                for (auto e = 0; e < 143; ++ e)
                {
                        NANO_CHECK_GREATER(nano::abs(iq.gradient(rnd(rng))), nano::abs(iq.gradient(extremum1)));
                        NANO_CHECK_GREATER(nano::abs(iq.gradient(rnd(rng))), nano::abs(iq.gradient(extremum2)));
                }
        }
}

NANO_END_MODULE()
