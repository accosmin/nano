#include "ncv.h"

// run loop for the given number of trials using no threads
template
<
        typename toperator
>
ncv::stats_t<ncv::scalar_t> test(int size, int trials, toperator op)
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

                ncv::thread_loop(size, op);

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
void print(const ncv::string_t& header,
           const ncv::stats_t<ncv::scalar_t>& timings,
           const ncv::stats_t<ncv::scalar_t>& timings_ncv)
{
        static const ncv::size_t col_size = 32;

        std::cout << ncv::text::resize(header, col_size)
                  << to_string(timings, col_size)
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
                        1000000
                      };

        // test for different problems size
        std::for_each(std::begin(sizes), std::end(sizes), [&] (int size)
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

                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats_t<ncv::scalar_t> timings = test(size, 16, op);
                std::cout << "sum = " << std::accumulate(std::begin(results), std::end(results), 0.0) << std::endl;

                std::fill(std::begin(results), std::end(results), 0.0);
                const ncv::stats_t<ncv::scalar_t> timings_ncv = test_ncv(size, 16, op);
                std::cout << "sum = " << std::accumulate(std::begin(results), std::end(results), 0.0) << std::endl;

                print("test1 [" + ncv::text::to_string(size) + "]",
                      timings, timings_ncv);
        });

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
