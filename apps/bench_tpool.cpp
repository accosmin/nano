#include <cmath>
#include "core/table.h"
#include "core/tpool.h"
#include "core/numeric.h"
#include "core/cmdline.h"
#include "core/measure.h"
#include <iostream>

using namespace nano;

namespace
{
        template <typename tvector>
        void op(const size_t i, tvector& vector)
        {
                const auto x = static_cast<double>(i);
                vector[i] = std::sin(x) + std::cos(x);
        }

        template <typename tvector>
        void st_op(tvector& vector)
        {
                for (size_t i = 0; i < vector.size(); ++ i)
                {
                        op(i, vector);
                }
                (void)vector;
        }

        template <typename tvector>
        void mt_op(tvector& vector)
        {
                nano::loopi(vector.size(), [&] (const size_t i)
                {
                        op(i, vector);
                });
                (void)vector;
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
        header << "function" << "st" << "mt";
        table.delim();

        // benchmark for different problem sizes and number of active workers
        for (size_t size = cmd_min_size; size <= cmd_max_size; size *= 2)
        {
                auto& row = table.append();
                row << ("sin+cos [" + to_string(size / kilo) + "K]");

                std::vector<double> vector0(size);
                const auto delta0 = measure<nanoseconds_t>([&] { st_op(vector0); }, 16);
                row << precision(2) << (static_cast<double>(delta0.count()) / static_cast<double>(delta0.count()));

                std::vector<double> vectorX(size);
                const auto deltaX = measure<nanoseconds_t>([&] { mt_op(vectorX); }, 16);

                row << precision(2) << (static_cast<double>(delta0.count()) / static_cast<double>(deltaX.count()));

                for (size_t i = 0; i < size; ++ i)
                {
                        if (std::fabs(vector0[i] - vectorX[i]) > 1e-16)
                        {
                                std::cerr << "mis-matching vector (i=" << i
                                        << ",delta=" << std::fabs(vector0[i] - vectorX[i]) << ")!" << std::endl;
                                return EXIT_FAILURE;
                        }
                }
        }

        // print results
        table.mark(make_marker_maximum_percentage_cols<double>(5));
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
