#include <iomanip>
#include <iostream>
#include "math/stats.h"
#include "text/table.h"
#include "text/cmdline.h"

using namespace nano;

static stats_t<double> get_stats(const table_t& table, const size_t col)
{
        stats_t<double> stats;
        for (size_t row = 0; row < table.rows(); ++ row)
        {
                try
                {
                        stats(from_string<double>(table.row(row).data(col)));
                }
                catch (std::exception&)
                {
                }
        }

        return stats;
}

int main(int argc, const char *argv[])
{
        // parse the command line
        nano::cmdline_t cmdline("tabulate a csv file");
        cmdline.add("i", "input",       "input path (.csv)");
        cmdline.add("d", "delim",       "delimiting character(s)");
        cmdline.add("p", "precision",   "precision for floating point values", 6);
        cmdline.add("s", "stats",       "print statistics for each column (except the first one)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_input = cmdline.get("input");
        const auto cmd_delim = cmdline.get("delim");
        const auto cmd_stats = cmdline.has("stats");
        const auto cmd_precision = cmdline.get<int>("precision");

        // tabulate
        table_t table;
        if (!table.load(cmd_input, cmd_delim))
        {
                std::cerr << "failed to load <" << cmd_input << ">!\n";
                return EXIT_FAILURE;
        }
        else
        {
                std::cout << std::fixed << std::setprecision(cmd_precision);

                if (!cmd_stats)
                {
                        std::cout << table;
                }
                else
                {
                        size_t row_header = 0;
                        for (size_t row = 0; row < table.rows(); ++ row)
                        {
                                if (table.row(row).type() == row_t::mode::header)
                                {
                                        row_header = row;
                                        break;
                                }
                        }

                        const auto& header = table.row(row_header);

                        // compute statistics for each column
                        table_t stats_table;
                        stats_table.header() << "" << "avg" << "var" << "stdev" << "min" << "max";
                        stats_table.delim();
                        for (size_t col = 1; col < table.cols(); ++ col)
                        {
                                const auto stats = get_stats(table, col);

                                auto&& row = stats_table.append();
                                row << header.data(col) << stats.avg() << stats.var() << stats.stdev()
                                    << stats.min() << stats.max();
                        }

                        std::cout << stats_table;
                }

                return EXIT_SUCCESS;
        }
}
