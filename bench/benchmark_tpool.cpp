#include <cmath>
#include "core/table.h"
#include "thread/loopi.h"
#include "core/numeric.h"
#include "core/cmdline.h"
#include "core/measure.h"
#include <iostream>

using namespace nano;

namespace
{
        template <typename tvector>
        void op(tvector& results)
        {
                nano::loopi(results.size(), size_t(1), [&results = results] (const auto begin, const auto end)
                {
                        for (auto i = begin; i < end; ++ i)
                        {
                                const auto x = static_cast<scalar_t>(i);
                                results[i] = std::sin(x) + std::cos(x);
                        }
                });
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
        auto& header = table.header();
        header << "function";
        for (size_t n_active_workers = 1; n_active_workers <= n_threads; ++ n_active_workers)
        {
                header << (to_string(n_active_workers) + (n_active_workers == 1 ? "thread" : "threads"));
        }
        table.delim();

        // benchmark for different problem sizes and number of active workers
        for (size_t size = cmd_min_size; size <= cmd_max_size; size *= 2)
        {
                auto& row = table.append();
                row << ("sin+cos [" + to_string(size / kilo) + "K]");

                microseconds_t delta1(0);

                for (size_t n_active_workers = 1; n_active_workers <= n_threads; ++ n_active_workers)
                {
                        std::vector<scalar_t> results(size);
                        const auto deltaX = measure<microseconds_t>([&] { op(results); }, 16);
                        if (n_active_workers == 1)
                        {
                                delta1 = deltaX;
                        }

                        row << (static_cast<scalar_t>(delta1.count()) / static_cast<scalar_t>(deltaX.count()));
                }
        }

        // print results
        table.mark(make_marker_maximum_percentage_cols<scalar_t>(5));
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
