#include "utest.h"
#include "math/stats.h"
#include "math/random.h"

namespace test
{
        void check_stats(double avg, double var, size_t count)
        {
                nano::stats_t<double> stats;
                auto rgen = nano::make_rng<double>(-var, +var);

                // generate random values
                std::vector<double> values;
                for (size_t i = 0; i < count; ++ i)
                {
                        values.push_back(avg + rgen());

                        stats(*values.rbegin());
                }

                // check count
                NANO_CHECK_EQUAL(stats.count(), count);

                // check range
                NANO_CHECK_GREATER_EQUAL(stats.min(), avg - var);
                NANO_CHECK_LESS_EQUAL(stats.max(), avg + var);

                // check average
                NANO_CHECK_GREATER_EQUAL(stats.avg(), avg - var);
                NANO_CHECK_LESS_EQUAL(stats.avg(), avg + var);

                // check variance
                NANO_CHECK_GREATER_EQUAL(stats.var(), 0.0);

                // check sum
                NANO_CHECK_CLOSE(stats.sum(), std::accumulate(values.begin(), values.end(), 0.0), 1e-8);
        }
}

NANO_BEGIN_MODULE(test_stats)

NANO_CASE(evaluate)
{
        test::check_stats(0.03, 0.005, 32);
        test::check_stats(1.03, 13.005, 37);
        test::check_stats(-0.54, 0.105, 13);
        test::check_stats(-7.03, 10.005, 11);
}

NANO_END_MODULE()

