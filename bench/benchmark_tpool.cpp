#include <cmath>
#include "core/table.h"
#include "core/loopi.h"
#include "core/numeric.h"
#include "core/cmdline.h"
#include "core/measure.h"
#include <iostream>

using namespace nano;

namespace
{
        template <typename tvector>
        void op(const size_t begin, const size_t end, tvector& results)
        {
                for (auto i = begin; i < end; ++ i)
                {
                        const auto x = static_cast<double>(i);
                        results[i] = std::sin(x) + std::cos(x);
                }
        }

        template <typename tvector>
        void st_op(tvector& results)
        {
                op(size_t(0), results.size(), results);
        }

        template <typename tvector>
        void mt_op(const size_t chunk, tvector& results)
        {
                nano::loopi(results.size(), chunk, [&] (const auto begin, const auto end) { op(begin, end, results); });
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

        table_t table;
        auto& header = table.header();
        header << "function" << "st" << "mt-c1" << "mt-c2" << "mt-c4" << "mt-c8" << "mt-c16" << "mt-c32" << "mt-c64" << "mt-c128";
        table.delim();

        // benchmark for different problem sizes and number of active workers
        for (size_t size = cmd_min_size; size <= cmd_max_size; size *= 2)
        {
                auto& row = table.append();
                row << ("sin+cos [" + to_string(size / kilo) + "K]");

                std::vector<double> results(size);

                std::fill(results.begin(), results.end(), 0);
                const auto delta0 = measure<nanoseconds_t>([&] { st_op(results); }, 16);
                row << precision(2) << (static_cast<double>(delta0.count()) / static_cast<double>(delta0.count()));

                for (size_t chunk = 1; chunk <= 128; chunk *= 2)
                {
                        std::fill(results.begin(), results.end(), 0);
                        const auto deltaX = measure<nanoseconds_t>([&] { mt_op(chunk, results); }, 16);

                        row << precision(2) << (static_cast<double>(delta0.count()) / static_cast<double>(deltaX.count()));
                }
        }

        // print results
        table.mark(make_marker_maximum_percentage_cols<double>(5));
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
