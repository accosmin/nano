#include "ncv_thread.h"
#include "ncv_logger.h"
#include "ncv_random.h"
#include "ncv_stats.h"
#include "ncv_timer.h"
#include "ncv_string.h"
#include <omp.h>

// Run loop for the given number of trials using no threads
template
<
        typename toperator
>
ncv::stats<> test(int size, int trials, const toperator& op)
{
        ncv::stats<> timings;
        for (int t = 0; t < trials; t ++)
        {
                const ncv::timer timer;

                for (int i = 0; i < size; i ++)
                {
                        op(i);
                }

                timings.add(timer.elapsed_miliseconds());
        }

        return timings;
}

// Run loop for the given number of trials using OpenMP
template
<
        typename toperator
>
ncv::stats<> test_omp(int size, int trials, const toperator& op)
{
        ncv::stats<> timings;
        for (int t = 0; t < trials; t ++)
        {
                const ncv::timer timer;

                #pragma omp parallel for
                for (int i = 0; i < size; i ++)
                {
                        op(i);
                }

                timings.add(timer.elapsed_miliseconds());
        }

        return timings;
}

// Run loop for the given number of trials using ncv::thread_loop
template
<
        typename toperator
>
ncv::stats<> test_ncv(int size, int trials, const toperator& op)
{
        ncv::stats<> timings;
        for (int t = 0; t < trials; t ++)
        {
                const ncv::timer timer;

                ncv::thread_loop(size, op);

                timings.add(timer.elapsed_miliseconds());
        }

        return timings;
}

ncv::string_t to_string(const ncv::stats<>& timings, ncv::index_t col_size)
{
        return  ncv::text::resize(
                ncv::text::to_string(timings.avg()) + " +/- " + ncv::text::to_string(timings.stdev()),
                col_size);
}

// Display the formatted timing statistics
void print(const ncv::string_t& header,
           const ncv::stats<>& timings,
           const ncv::stats<>& timings_omp,
           const ncv::stats<>& timings_ncv)
{
        static const ncv::index_t col_size = 32;

        std::cout << ncv::text::resize(header, col_size)
                  << to_string(timings, col_size)
                  << to_string(timings_omp, col_size)
                  << to_string(timings_ncv, col_size)
                  << std::endl;
}

int main(int argc, char *argv[])
{
        int sizes[] = {
                        10,
                        100,
                        1000,
                        10000,
                        100000,
                        1000000,
                        10000000,
                        100000000
                      };

        // Test for different problems size
        std::for_each(std::begin(sizes), std::end(sizes), [&] (int size)
        {
                ncv::scalars_t results(size);

                // Operator for test
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

                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats<> timings     = test    (size, 16, op);
                std::cout << "sum = " << std::accumulate(std::begin(results), std::end(results), 0.0) << std::endl;

                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats<> timings_omp = test_omp(size, 16, op);
                std::cout << "sum = " << std::accumulate(std::begin(results), std::end(results), 0.0) << std::endl;

                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats<> timings_ncv = test_ncv(size, 16, op);
                std::cout << "sum = " << std::accumulate(std::begin(results), std::end(results), 0.0) << std::endl;

                print("test1 [" + ncv::text::to_string(size) + "]",
                      timings, timings_omp, timings_ncv);
        });

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
