#include <cmath>
#include "timer.h"
#include "text/table.h"
#include "thread/loopi.h"
#include "math/numeric.h"
#include "text/cmdline.h"
#include "text/table_row_mark.h"
#include <iostream>

using namespace nano;

namespace
{
        template <typename tindex>
        scalar_t op(const tindex index)
        {
                const auto x = static_cast<scalar_t>(index);
                return std::sin(x) + std::cos(x);
        }

        template <typename tvector>
        auto measure(tvector& results)
        {
                const nano::timer_t timer;
                nano::loopi(results.size(), [&results = results] (const auto index) { results[index] = op(index); });
                return timer.microseconds();
        }
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark thread pool");
        cmdline.add("", "min-size",     "minimum problem size (in kilo)", "1");
        cmdline.add("", "max-size",     "maximum problem size (in kilo)", "1024");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto kilo = size_t(1024);
        const auto cmd_min_size = clamp(kilo * cmdline.get<size_t>("min-size"), kilo, 1024 * kilo);
        const auto cmd_max_size = clamp(kilo * cmdline.get<size_t>("max-size"), cmd_min_size, 1024 * 1024 * kilo);

        const size_t n_threads = nano::logical_cpus();

        table_t table;
        table.header() << "function";
        for (size_t n_active_workers = 1; n_active_workers <= n_threads; ++ n_active_workers)
        {
                table.header() << (to_string(n_active_workers) + " thread(s)");
        }

        // benchmark for different problem sizes and number of active workers
        for (size_t size = cmd_min_size; size <= cmd_max_size; size *= 2)
        {
                auto& row = table.append();
                row << ("sin+cos [" + to_string(size / kilo) + "K]");

                microseconds_t duration1(0);

                for (size_t n_active_workers = 1; n_active_workers <= n_threads; ++ n_active_workers)
                {
                        auto& pool = thread_pool_t::instance();
                        pool.activate(n_active_workers);

                        std::vector<scalar_t> results(size);
                        const auto durationx = ::measure(results);
                        if (n_active_workers == 1)
                        {
                                duration1 = durationx;
                        }

                        const auto scaling = static_cast<scalar_t>(duration1.count()) / static_cast<scalar_t>(durationx.count());
                        row << scaling;
                }
        }

        // print results
        table.mark(make_table_mark_maximum_percentage_cols<size_t>(5));
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
