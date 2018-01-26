#include <numeric>
#include <iomanip>
#include <iostream>
#include "math/stats.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "text/algorithm.h"

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
        cmdline.add("c", "cols",        "use these colums to print statistics (e.g. 1,2,3)", "-");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_input = cmdline.get("input");
        const auto cmd_delim = cmdline.get("delim");
        const auto cmd_stats = cmdline.has("stats");
        const auto cmd_precision = cmdline.get<int>("precision");
        const auto cmd_cols = cmdline.get("cols");

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
                        // select columns
                        std::vector<size_t> cols;
                        if (cmd_cols.empty() || cmd_cols == "-")
                        {
                                cols.resize(table.cols() - 1);
                                std::iota(cols.begin(), cols.end(), size_t(1));
                        }
                        else
                        {
                                const auto tokens = split(cmd_cols, ",");
                                if (tokens.empty())
                                {
                                        std::cerr << "invalid columns!\n";
                                        return EXIT_FAILURE;
                                }

                                for (const auto& token : tokens)
                                {
                                        try
                                        {
                                                cols.push_back(from_string<size_t>(token));
                                        }
                                        catch (std::exception&)
                                        {
                                                std::cerr << "invalid column <" << token << "!>\n";
                                                return EXIT_FAILURE;
                                        }
                                }
                        }

                        // compute statistics for each of the selected columns
                        for (const auto col : cols)
                        {
                                const auto stats = get_stats(table, col);
                                std::cout << stats << std::endl;
                        }
                }

                return EXIT_SUCCESS;
        }
}
