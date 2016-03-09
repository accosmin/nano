#include "unit_test.hpp"
#include "math/stats.hpp"
#include "math/random.hpp"

namespace test
{
        void check_stats(double avg, double var, size_t count)
        {
                zob::stats_t<double> stats;
                auto rgen = zob::make_rng<double>(-var, +var);

                // generate random values
                std::vector<double> values;
                for (size_t i = 0; i < count; ++ i)
                {
                        values.push_back(avg + rgen());

                        stats(*values.rbegin());
                }

                // check count
                ZOB_CHECK_EQUAL(stats.count(), count);

                // check range
                ZOB_CHECK_GREATER_EQUAL(stats.min(), avg - var);
                ZOB_CHECK_LESS_EQUAL(stats.max(), avg + var);

                // check average
                ZOB_CHECK_GREATER_EQUAL(stats.avg(), avg - var);
                ZOB_CHECK_LESS_EQUAL(stats.avg(), avg + var);

                // check variance
                ZOB_CHECK_GREATER_EQUAL(stats.var(), 0.0);
                ZOB_CHECK_LESS_EQUAL(std::sqrt(stats.var()), var);

                // check sum
                ZOB_CHECK_CLOSE(stats.sum(), std::accumulate(values.begin(), values.end(), 0.0), 1e-8);
        }
}

ZOB_BEGIN_MODULE(test_stats)

ZOB_CASE(evaluate)
{
        test::check_stats(0.03, 0.005, 32);
        test::check_stats(1.03, 13.005, 37);
        test::check_stats(-0.54, 0.105, 13);
        test::check_stats(-7.03, 10.005, 11);
}

ZOB_END_MODULE()

