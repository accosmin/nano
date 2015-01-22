#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_stats"

#include <boost/test/unit_test.hpp>
#include <numeric>
#include "util/stats.hpp"
#include "util/random.hpp"

namespace test
{
        using namespace ncv;

        void check_stats(double avg, double var, size_t count)
        {
                ncv::stats_t<double> stats;
                ncv::random_t<double> rgen(-var, +var);

                // generate random values
                std::vector<double> values;
                for (size_t i = 0; i < count; i ++)
                {
                        values.push_back(avg + rgen());

                        stats(*values.rbegin());
                }

                // check count
                BOOST_CHECK_EQUAL(stats.count(), count);

                // check range
                BOOST_CHECK_GE(stats.min(), avg - var);
                BOOST_CHECK_LE(stats.max(), avg + var);

                // check average
                BOOST_CHECK_GE(stats.avg(), avg - var);
                BOOST_CHECK_LE(stats.avg(), avg + var);

                // check variance
                BOOST_CHECK_GE(stats.var(), 0.0);
                BOOST_CHECK_LE(std::sqrt(stats.var()), var);

                // check sum
                BOOST_CHECK_LE(std::fabs(stats.sum() - std::accumulate(values.begin(), values.end(), 0.0)), 1e-8);
        }
}

BOOST_AUTO_TEST_CASE(test_stats)
{
        test::check_stats(0.03, 0.005, 32);
        test::check_stats(1.03, 13.005, 37);
        test::check_stats(-0.54, 0.105, 13);
        test::check_stats(-7.03, 10.005, 11);
}
