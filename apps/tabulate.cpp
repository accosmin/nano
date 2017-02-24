#include "text/table.h"
#include "text/cmdline.h"
#include <iostream>

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("tabulate a csv file");
        cmdline.add("i", "input",       "input path (.csv)");
        cmdline.add("d", "delim",       "delimiting character(s)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_input = cmdline.get("input");
        const auto cmd_delim = cmdline.get("delim");

        // tabulate
        table_t table("");
        if (!table.load(cmd_input, cmd_delim))
        {
                std::cerr << "failed to load <" << cmd_input << ">!\n";
                return EXIT_FAILURE;
        }
        else
        {
                std::cout << table;
                return EXIT_SUCCESS;
        }
}
