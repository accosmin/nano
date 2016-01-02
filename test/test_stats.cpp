#include "unit_test.hpp"
#include "math/stats.hpp"
#include "math/random.hpp"

namespace test
{
        void check_stats(double avg, double var, size_t count)
        {
                math::stats_t<double> stats;
                auto rgen = math::make_rng<double>(-var, +var);

                // generate random values
                std::vector<double> values;
                for (size_t i = 0; i < count; ++ i)
                {
                        values.push_back(avg + rgen());

                        stats(*values.rbegin());
                }

                // check count
                NANOCV_CHECK_EQUAL(stats.count(), count);

                // check range
                NANOCV_CHECK_GREATER_EQUAL(stats.min(), avg - var);
                NANOCV_CHECK_LESS_EQUAL(stats.max(), avg + var);

                // check average
                NANOCV_CHECK_GREATER_EQUAL(stats.avg(), avg - var);
                NANOCV_CHECK_LESS_EQUAL(stats.avg(), avg + var);

                // check variance
                NANOCV_CHECK_GREATER_EQUAL(stats.var(), 0.0);
                NANOCV_CHECK_LESS_EQUAL(std::sqrt(stats.var()), var);

                // check sum
                NANOCV_CHECK_CLOSE(stats.sum(), std::accumulate(values.begin(), values.end(), 0.0), 1e-8);
        }
}

NANOCV_BEGIN_MODULE(test_stats)

NANOCV_CASE(evaluate)
{
        test::check_stats(0.03, 0.005, 32);
        test::check_stats(1.03, 13.005, 37);
        test::check_stats(-0.54, 0.105, 13);
        test::check_stats(-7.03, 10.005, 11);
}

NANOCV_END_MODULE()

