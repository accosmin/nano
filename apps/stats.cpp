#include <iomanip>
#include <iostream>
#include "core/cast.h"
#include "core/stats.h"

using namespace nano;

int main(int argc, const char* argv[])
{
        if (argc == 1)
        {
                std::cout << "usage: [-p (precision, default 6)] <list of numbers>\n";
                return EXIT_FAILURE;
        }

        try
        {
                int precision = 6;

                stats_t<double> stats;
                for (int i = 1; i < argc; ++ i)
                {
                        if (std::string(argv[i]) == "-p" && i + 1 < argc)
                        {
                                precision = from_string<int>(argv[i + 1]);
                                ++ i;
                        }
                        else
                        {
                                stats(from_string<double>(argv[i]));
                        }
                }

                std::cout << std::fixed << std::setprecision(precision) << stats << "\n";
        }
        catch (std::exception& e)
        {
                std::cout << "failed with error <" << e.what() << ">, check arguments!\n";
                return EXIT_FAILURE;
        }

        // OK
        return EXIT_SUCCESS;
}
