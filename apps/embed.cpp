#include "stringi.h"
#include "io/buffer.h"
#include "text/cmdline.h"
#include "cortex/logger.h"
#include "text/algorithm.h"
#include "text/filesystem.h"
#include <fstream>

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("embed binary file into the library");
        cmdline.add("i", "input",       "input path");
        cmdline.add("o", "output",      "output base file name (to generate .h & .cpp)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_input = cmdline.get<string_t>("input");
        const auto cmd_output = cmdline.get<string_t>("output");

        // load input file
        nano::buffer_t data;
        if (!nano::load_buffer(cmd_input, data))
        {
                log_error() << "failed to load input file from <" << cmd_input << ">!";
                return EXIT_FAILURE;
        }

        log_info () << "input: " << data.size() << " bytes.";

        const string_t name = nano::lower(nano::replace(nano::stem(cmd_input), '-', '_'));
        const string_t tab(8, ' ');
        const size_t rowsize = 128;

        const string_t path_header = nano::lower(nano::replace(cmd_output, '-', '_')) + ".h";
        const string_t path_source = nano::lower(nano::replace(cmd_output, '-', '_')) + ".cpp";

        // generate header
        std::ofstream os_header(path_header.c_str(), std::ios::out);
        if (!os_header.is_open())
        {
                log_error() << "failed to open header <" << path_header << ">!";
                return EXIT_FAILURE;
        }

        os_header << "#pragma once\n";
        os_header << "\n";
        os_header << "#include <cstddef>\n";
        os_header << "\n";
        os_header << "namespace nano\n";
        os_header << "{\n";
        os_header << tab << "const char* get_" << name << "_data();\n";
        os_header << tab << "std::size_t get_" << name << "_size();\n";
        os_header << tab << "const char* get_" << name << "_name();\n";
        os_header << "}\n";
        os_header << "\n";

        if (!os_header.good())
        {
                log_error() << "failed to write header <" << path_header << ">!";
                return EXIT_FAILURE;
        }
        os_header.close();

        // generate source
        std::ofstream os_source(path_source.c_str(), std::ios::out);
        if (!os_source.is_open())
        {
                log_error() << "failed to open source <" << path_source << ">!";
                return EXIT_FAILURE;
        }

        os_source << "#include " << nano::filename(path_header) << "\n";
        os_source << "\n";
        os_source << "namespace\n";
        os_source << "{\n";
        os_source << tab << "constexpr char name[] = \"" << name << "\";\n\n";
        os_source << tab << "constexpr unsigned char data[] = \n";
        os_source << tab << "{\n" << tab << tab;
        for (size_t i = 0; i < data.size(); i ++)
        {
                os_source << (data[i] & 0xFF);
                if (i + 1 < data.size())
                {
                        os_source << ",";
                        if (i && i % rowsize == 0)
                        {
                                os_source << "\n" << tab << tab;
                        }
                }
        }
        os_source << "\n" << tab << "};\n";
        os_source << "}\n\n";

        os_source << "const char* nano::get_" << name << "_data()\n";
        os_source << "{\n";
        os_source << tab << "return (const char*)data;\n";
        os_source << "}\n\n";

        os_source << "std::size_t nano::get_" << name << "_size()\n";
        os_source << "{\n";
        os_source << tab << "return sizeof(data) / sizeof(unsigned char);\n";
        os_source << "}\n\n";

        os_source << "const char* nano::get_" << name << "_name()\n";
        os_source << "{\n";
        os_source << tab << "return name;\n";
        os_source << "}\n\n";

        if (!os_source.good())
        {
                log_error() << "failed to write source <" << path_source << ">!";
                return EXIT_FAILURE;
        }
        os_source.close();

        // OK
        log_info() << nano::done;
        return EXIT_SUCCESS;
}
