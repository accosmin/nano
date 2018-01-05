#include "utest.h"
#include "math/stats.h"
#include "math/random.h"

using namespace nano;

NANO_BEGIN_MODULE(test_stats)

NANO_CASE(fixed)
{
        stats_t<double> stats;
        stats(2, 4, 4, 4, 5, 5, 7, 9);

        NANO_CHECK_EQUAL(stats.count(), size_t(8));
        NANO_CHECK_EQUAL(stats.min(), 2.0);
        NANO_CHECK_EQUAL(stats.max(), 9.0);
        NANO_CHECK_CLOSE(stats.sum1(), 40.0, 1e-16);
        NANO_CHECK_CLOSE(stats.sum2(), 232.0, 1e-16);
        NANO_CHECK_CLOSE(stats.var(), 4.0, 1e-16);
        NANO_CHECK_CLOSE(stats.stdev(), 2.0, 1e-16);
}

NANO_CASE(merge)
{
        stats_t<double> stats1;
        stats1(2, 4, 4);

        stats_t<double> stats2;
        stats2(4, 5, 5, 7, 9);

        stats_t<double> stats;
        stats(stats1);
        stats(stats2);

        NANO_CHECK_EQUAL(stats.count(), size_t(8));
        NANO_CHECK_EQUAL(stats.min(), 2.0);
        NANO_CHECK_EQUAL(stats.max(), 9.0);
        NANO_CHECK_CLOSE(stats.sum1(), 40.0, 1e-16);
        NANO_CHECK_CLOSE(stats.sum2(), 232.0, 1e-16);
        NANO_CHECK_CLOSE(stats.var(), 4.0, 1e-16);
        NANO_CHECK_CLOSE(stats.stdev(), 2.0, 1e-16);
}

NANO_CASE(random)
{
        const auto avg = -4.2;
        const auto var = 0.47;
        const auto count = size_t(37);

        auto rgen = nano::make_rng<double>(-var, +var);

        // generate random values
        std::vector<double> values;
        for (size_t i = 0; i < count; ++ i)
        {
                values.push_back(avg + rgen());
        }

        const auto min = *std::min_element(values.begin(), values.end());
        const auto max = *std::max_element(values.begin(), values.end());

        const auto sum1 = std::accumulate(values.begin(), values.end(), 0.0);
        const auto sum2 = std::accumulate(values.begin(), values.end(), 0.0,
                [] (const auto acc, const auto val) { return acc + val * val; });

        stats_t<double> stats;
        stats(values.begin(), values.end());

        NANO_CHECK_EQUAL(stats.count(), count);
        NANO_CHECK_CLOSE(stats.min(), min, 1e-16);
        NANO_CHECK_CLOSE(stats.max(), max, 1e-16);
        NANO_CHECK_CLOSE(stats.sum1(), sum1, 1e-12);
        NANO_CHECK_CLOSE(stats.sum2(), sum2, 1e-12);

        NANO_CHECK_LESS_EQUAL(stats.max(), avg + var);
        NANO_CHECK_GREATER_EQUAL(stats.min(), avg - var);

        NANO_CHECK_CLOSE(stats.avg(), sum1 / count, 1e-12);
        NANO_CHECK_LESS_EQUAL(stats.avg(), avg + var);
        NANO_CHECK_GREATER_EQUAL(stats.avg(), avg - var);

        NANO_CHECK_GREATER_EQUAL(stats.var(), 0.0);
        NANO_CHECK_CLOSE(stats.var(), (sum2 - sum1 * sum1 / count) / count, 1e-12);
}

NANO_END_MODULE()
