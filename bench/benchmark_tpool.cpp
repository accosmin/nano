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
        void op(const size_t chunk, tvector& results)
        {
                nano::loopi(results.size(), chunk, [&results = results] (const auto begin, const auto end)
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

        table_t table;
        auto& header = table.header();
        header << "function" << "chunk1" << "chunk2" << "chunk4" << "chunk8" << "chunk16" << "chunk32" << "chunk64" << "chunk128";
        table.delim();

        // benchmark for different problem sizes and number of active workers
        for (size_t size = cmd_min_size; size <= cmd_max_size; size *= 2)
        {
                auto& row = table.append();
                row << ("sin+cos [" + to_string(size / kilo) + "K]");

                microseconds_t delta1(0);

                for (size_t chunk = 1; chunk <= 128; chunk *= 2)
                {
                        std::vector<scalar_t> results(size);
                        const auto deltaX = measure<microseconds_t>([&] { op(chunk, results); }, 16);
                        if (chunk == 1)
                        {
                                delta1 = deltaX;
                        }

                        row << nano::precision(3) << (static_cast<scalar_t>(delta1.count()) / static_cast<scalar_t>(deltaX.count()));
                }
        }

        // print results
        table.mark(make_marker_maximum_percentage_cols<scalar_t>(5));
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
