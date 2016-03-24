#include <fstream>
#include "unit_test.hpp"
#include "text/cmdline.h"

NANO_BEGIN_MODULE(test_cmdline)

NANO_CASE(parse)
{
        nano::cmdline_t cmdline("unit testing");
        cmdline.add("v", "version", "version", "0.3");
        cmdline.add("", "iterations", "number of iterations", "127");

        const int argc = 4;
        const char* argv[] = { "", "-v", "--iterations", "7" };

        cmdline.process(argc, argv);

        NANO_CHECK(cmdline.has("v"));
        NANO_CHECK(cmdline.has("version"));
        NANO_CHECK(cmdline.has("iterations"));
        NANO_CHECK(!cmdline.has("h"));
        NANO_CHECK(!cmdline.has("help"));

        NANO_CHECK_EQUAL(cmdline.get<std::string>("v"), "0.3");
        NANO_CHECK_EQUAL(cmdline.get<int>("iterations"), 7);
}
/*
NANO_CASE(error_invalid_arg)
{
        nano::cmdline_t cmdline("unit testing");
        cmdline.add("v", "version", "version");
        cmdline.add("", "iterations", "number of iterations", "127");

        const int argc = 4;
        const char* argv[] = { "", "v", "--version", "7" };

        NANO_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

NANO_CASE(error_unknown_arg)
{
        nano::cmdline_t cmdline("unit testing");
        cmdline.add("v", "version", "version");
        cmdline.add("", "iterations", "number of iterations", "127");

        const int argc = 4;
        const char* argv[] = { "", "-v", "--what", "7" };

        NANO_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}
*/
NANO_CASE(parse_config_file)
{
        nano::cmdline_t cmdline("unit testing");
        cmdline.add("v", "version", "version", "0.3");
        cmdline.add("", "iterations", "number of iterations", "127");

        const std::string path = "config";

        {
                std::ofstream out(path.c_str());
                out << "-v\n";
                out << "--iterations 29";
        }

        cmdline.process_config_file(path);

        NANO_CHECK(cmdline.has("v"));
        NANO_CHECK(cmdline.has("version"));
        NANO_CHECK(cmdline.has("iterations"));
        NANO_CHECK(!cmdline.has("h"));
        NANO_CHECK(!cmdline.has("help"));

        NANO_CHECK_EQUAL(cmdline.get<std::string>("v"), "0.3");
        NANO_CHECK_EQUAL(cmdline.get<int>("iterations"), 29);

        std::remove(path.c_str());
}

NANO_END_MODULE()
