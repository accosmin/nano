#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_thread_loop"

#include <boost/test/unit_test.hpp>
#include "nanocv.h"

namespace test
{
        // run loop for the given number of trials using no threads
        template
        <
                typename toperator
        >
        ncv::stats_t<ncv::scalar_t> test_cpu(int size, int trials, toperator op)
        {
                ncv::stats_t<ncv::scalar_t> timings;
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
        ncv::stats_t<ncv::scalar_t> test_omp(int size, int trials, toperator op)
        {
                ncv::stats_t<ncv::scalar_t> timings;
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

        // run loop for the given number of trials using ncv::thread_loop
        template
        <
                typename toperator
        >
        ncv::stats_t<ncv::scalar_t> test_ncv(int size, int trials, toperator op)
        {
                ncv::stats_t<ncv::scalar_t> timings;
                for (int t = 0; t < trials; t ++)
                {
                        const ncv::timer_t timer;

                        ncv::thread_loopi(size, op);

                        timings(timer.miliseconds());
                }

                return timings;
        }

        // run loop for the given number of trials using ncv::thread_loop
        template
        <
                typename toperator
        >
        ncv::stats_t<ncv::scalar_t> test_ncv_pool(int size, int trials, toperator op)
        {
                static ncv::thread_pool_t pool;

                ncv::stats_t<ncv::scalar_t> timings;
                for (int t = 0; t < trials; t ++)
                {
                        const ncv::timer_t timer;

                        ncv::thread_loopi(size, pool, op);

                        timings(timer.miliseconds());
                }

                return timings;
        }

        ncv::string_t to_string(const ncv::stats_t<ncv::scalar_t>& timings, ncv::size_t col_size)
        {
                return  ncv::text::resize(
                        ncv::text::to_string(timings.avg()) + " +/- " + ncv::text::to_string(timings.stdev()),
                        col_size);
        }

        // display the formatted timing statistics
        void print(const ncv::string_t& header, size_t col_size,
                   const ncv::stats_t<ncv::scalar_t>& timings_cpu,
#ifdef _OPENMP
                   const ncv::stats_t<ncv::scalar_t>& timings_omp,
#endif
                   const ncv::stats_t<ncv::scalar_t>& timings_ncv,
                   const ncv::stats_t<ncv::scalar_t>& timings_ncv_pool)
        {
                std::cout << ncv::text::resize(header, col_size)
                          << to_string(timings_cpu, col_size)
#ifdef _OPENMP
                          << to_string(timings_omp, col_size)
#endif
                          << to_string(timings_ncv, col_size)
                          << to_string(timings_ncv_pool, col_size)
                          << std::endl;
        }
}

BOOST_AUTO_TEST_CASE(test_thread_loop)
{
        using namespace ncv;

        const size_t min_size = 128;
        const size_t max_size = 1024 * 1024;

        const size_t trials = 16;
        const size_t col_size = 28;

        // test for different problems size
        std::cout << ncv::text::resize("", col_size)
                  << ncv::text::resize("CPU", col_size)
#ifdef _OPENMP
                  << ncv::text::resize("OpenMP", col_size)
#endif
                  << ncv::text::resize("NanoCV", col_size)
                  << ncv::text::resize("NanoCV(pool)", col_size)
                  << std::endl;

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                ncv::scalars_t results(size);

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
                const ncv::stats_t<ncv::scalar_t> timings_cpu = test::test_cpu(size, trials, op);
                const scalar_t sum_cpu = std::accumulate(std::begin(results), std::end(results), 0.0);

#ifdef _OPENMP
                // OpenMP
                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats_t<ncv::scalar_t> timings_omp = test::test_omp(size, trials, op);
                const scalar_t sum_omp = std::accumulate(std::begin(results), std::end(results), 0.0);
#endif

                // NanoCV threads
                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats_t<ncv::scalar_t> timings_ncv = test::test_ncv(size, trials, op);
                const scalar_t sum_ncv = std::accumulate(std::begin(results), std::end(results), 0.0);

                // NanoCV threads with reusing the thread pool
                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats_t<ncv::scalar_t> timings_ncv_pool = test::test_ncv_pool(size, trials, op);
                const scalar_t sum_ncv_loop = std::accumulate(std::begin(results), std::end(results), 0.0);

                test::print("test [" + ncv::text::to_string(size) + "]", col_size,
                      timings_cpu,
#ifdef _OPENMP
                      timings_omp,
#endif
                      timings_ncv,
                      timings_ncv_pool);

                // Check accuracy
                const scalar_t eps = 1e-12;
                BOOST_CHECK_LE(std::fabs(sum_cpu - sum_cpu), eps);
#ifdef _OPENMP
                BOOST_CHECK_LE(std::fabs(sum_cpu - sum_omp), eps);
#endif
                BOOST_CHECK_LE(std::fabs(sum_cpu - sum_ncv), eps);
                BOOST_CHECK_LE(std::fabs(sum_cpu - sum_ncv_loop),  eps);
        }
}
