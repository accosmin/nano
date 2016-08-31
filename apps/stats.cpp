#include "math/stats.hpp"
#include "text/from_string.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, const char* argv[])
{
        if (argc == 1)
        {
                std::cout << "usage: <list of numbers>\n";
                return EXIT_FAILURE;
        }

        nano::stats_t<double> stats;
        for (int i = 1; i < argc; ++ i)
        {
                stats(nano::from_string<double>(argv[i]));
        }

        std::cout << std::fixed << std::setprecision(6) << stats << "\n";

        // OK
        return EXIT_SUCCESS;
}
