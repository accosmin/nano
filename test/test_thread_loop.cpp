#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_thread_loop"

#include <boost/test/unit_test.hpp>
#include "libnanocv/types.h"
#include "libnanocv/math/abs.hpp"
#include "libnanocv/util/timer.h"
#include "libnanocv/util/stats.hpp"
#include "libnanocv/util/tabulator.h"
#include "libnanocv/thread/parallel.hpp"
#include <iostream>
#include <numeric>

namespace test
{
        using namespace ncv;

        // run loop for the given number of trials using no threads
        template
        <
                typename toperator
        >
        stats_t<scalar_t> test_cpu(int size, int trials, toperator op)
        {
                stats_t<scalar_t> timings;
                for (int t = 0; t < trials; t ++)
                {
                        const ncv::timer_t timer;

                        for (int i = 0; i < size; i ++)
                        {
                                op(i);
                        }

                        timings(timer.miliseconds());
                }

                return timings;
        }

#ifdef _OPENMP
        // run loop for the given number of trials using OpenMP
        template
        <
                typename toperator
        >
        stats_t<scalar_t> test_omp(int size, int trials, toperator op)
        {
                stats_t<scalar_t> timings;
                for (int t = 0; t < trials; t ++)
                {
                        const ncv::timer_t timer;

                        #pragma omp parallel for schedule(static)
                        for (int i = 0; i < size; i ++)
                        {
                                op(i);
                        }

                        timings(timer.miliseconds());
                }

                return timings;
        }
#endif

        // run loop for the given number of trials using thread_loop
        template
        <
                typename toperator
        >
        stats_t<scalar_t> test_ncv(int size, int trials, toperator op)
        {
                stats_t<scalar_t> timings;
                for (int t = 0; t < trials; t ++)
                {
                        const ncv::timer_t timer;

                        thread_loopi(size, op);

                        timings(timer.miliseconds());
                }

                return timings;
        }

        // run loop for the given number of trials using thread_loop
        template
        <
                typename toperator
        >
        stats_t<scalar_t> test_ncv_pool(int size, int trials, toperator op)
        {
                static thread_pool_t pool;

                stats_t<scalar_t> timings;
                for (int t = 0; t < trials; t ++)
                {
                        const ncv::timer_t timer;

                        thread_loopi(size, pool, op);

                        timings(timer.miliseconds());
                }

                return timings;
        }

        string_t to_string(const stats_t<scalar_t>& timings)
        {
                return text::to_string(timings.avg()) + " +/- " + text::to_string(timings.stdev());
        }
}

BOOST_AUTO_TEST_CASE(test_thread_loop)
{
        using namespace ncv;

        const size_t min_size = 37;
        const size_t max_size = 256 * 1024;

        const size_t trials = 16;

        tabulator_t table("method");
        table.header() << "cpu";
#ifdef _OPENMP
        table.header() << "openmp";
#endif
        table.header() << "nanocv";
        table.header() << "nanocv(pool)";

        // test for different problems size
        for (size_t size = min_size; size <= max_size; size *= 3)
        {
                scalars_t results(size);

                // operator to test
                auto op = [&](int i)
                {
                        results[i] = std::cos(i + 0.0) + std::sin(i + 0.0);

                        double temp = 0.0;
                        for (int j = 0; j < 16; j ++)
                        {
                                temp += j * std::tan(-i + j);
                        }

                        results[i] *= temp / 32.0;
                };

                // 1CPU
                std::fill(std::begin(results), std::end(results), 0.0);
                const stats_t<scalar_t> timings_cpu = test::test_cpu(size, trials, op);
                const scalar_t sum_cpu = std::accumulate(std::begin(results), std::end(results), 0.0);

#ifdef _OPENMP
                // OpenMP
                std::fill(std::begin(results), std::end(results), 0.0);
                const stats_t<scalar_t> timings_omp = test::test_omp(size, trials, op);
                const scalar_t sum_omp = std::accumulate(std::begin(results), std::end(results), 0.0);
#endif

                // NanoCV threads
                std::fill(std::begin(results), std::end(results), 0.0);
                const stats_t<scalar_t> timings_ncv = test::test_ncv(size, trials, op);
                const scalar_t sum_ncv = std::accumulate(std::begin(results), std::end(results), 0.0);

                // NanoCV threads with reusing the thread pool
                std::fill(std::begin(results), std::end(results), 0.0);
                const stats_t<scalar_t> timings_ncv_pool = test::test_ncv_pool(size, trials, op);
                const scalar_t sum_ncv_loop = std::accumulate(std::begin(results), std::end(results), 0.0);

                table.append("test [" + text::to_string(size) + "]")
                << test::to_string(timings_cpu)
#ifdef _OPENMP
                << test::to_string(timings_omp)
#endif
                << test::to_string(timings_ncv)
                << test::to_string(timings_ncv_pool);

                // Check accuracy
                const scalar_t eps = 1e-12;
                BOOST_CHECK_LE(math::abs(sum_cpu - sum_cpu), eps);
#ifdef _OPENMP
                BOOST_CHECK_LE(math::abs(sum_cpu - sum_omp), eps);
#endif
                BOOST_CHECK_LE(math::abs(sum_cpu - sum_ncv), eps);
                BOOST_CHECK_LE(math::abs(sum_cpu - sum_ncv_loop),  eps);
        }

        table.print(std::cout);
}
