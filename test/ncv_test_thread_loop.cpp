#include "nanocv.h"

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
           const ncv::stats_t<ncv::scalar_t>& timings_omp,
           const ncv::stats_t<ncv::scalar_t>& timings_ncv,
           const ncv::stats_t<ncv::scalar_t>& timings_ncv_pool)
{
        std::cout << ncv::text::resize(header, col_size)
                  << to_string(timings_cpu, col_size)
                  << to_string(timings_omp, col_size)
                  << to_string(timings_ncv, col_size)
                  << to_string(timings_ncv_pool, col_size)
                  << std::endl;
}

int main(int argc, char *argv[])
{
        using namespace ncv;

        const size_t min_size = 128;
        const size_t max_size = 4 * 1024 * 1024;

        const size_t trials = 16;
        const size_t col_size = 28;

        // test for different problems size
        std::cout << ncv::text::resize("", col_size)
                  << ncv::text::resize("CPU", col_size)
                  << ncv::text::resize("OpenMP", col_size)
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
                const ncv::stats_t<ncv::scalar_t> timings_cpu = test_cpu(size, trials, op);
                const scalar_t sum_cpu = std::accumulate(std::begin(results), std::end(results), 0.0);

                // OpenMP
                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats_t<ncv::scalar_t> timings_omp = test_omp(size, trials, op);
                const scalar_t sum_omp = std::accumulate(std::begin(results), std::end(results), 0.0);

                // NanoCV threads
                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats_t<ncv::scalar_t> timings_ncv = test_ncv(size, trials, op);
                const scalar_t sum_ncv = std::accumulate(std::begin(results), std::end(results), 0.0);

                // NanoCV threads with reusing the thread pool
                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats_t<ncv::scalar_t> timings_ncv_pool = test_ncv_pool(size, trials, op);
                const scalar_t sum_ncv_loop = std::accumulate(std::begin(results), std::end(results), 0.0);

                print("test1 [" + ncv::text::to_string(size) + "]", col_size,
                      timings_cpu, timings_omp, timings_ncv, timings_ncv_pool);

                // Check accuracy
                const scalar_t eps = 1e-12;
                if (std::fabs(sum_cpu - sum_cpu) > eps)
                {
                        std::cout << "Error: [CPU] not valid! (" << sum_cpu << "/" << sum_cpu << ")" << std::endl;
                }
                if (std::fabs(sum_cpu - sum_omp) > eps)
                {
                        std::cout << "Error: [OpenMP] not valid! (" << sum_omp << "/" << sum_cpu << ")" << std::endl;
                }
                if (std::fabs(sum_cpu - sum_ncv) > eps)
                {
                        std::cout << "Error: [NanoCV] not valid! (" << sum_ncv << "/" << sum_cpu << ")" << std::endl;
                }
                if (std::fabs(sum_cpu - sum_ncv_loop) > eps)
                {
                        std::cout << "Error: [NanoCV(pool)] not valid! (" << sum_ncv_loop << "/" << sum_cpu << ")" << std::endl;
                }
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
