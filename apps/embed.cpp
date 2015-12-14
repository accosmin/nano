#include "io/buffer.h"
#include "cortex/string.h"
#include "text/algorithm.h"
#include "cortex/util/logger.h"
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        using namespace cortex;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "embed binary file into the library");
        po_desc.add_options()("input,i",
                boost::program_options::value<cortex::string_t>(),
                "input path");
        po_desc.add_options()("output,o",
                boost::program_options::value<cortex::string_t>(),
                "output base file name (to generate .h & .cpp)");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);

        // check arguments and options
        if (po_vm.empty() ||
                !po_vm.count("input") ||
                !po_vm.count("output") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_input = po_vm["input"].as<string_t>();
        const string_t cmd_output = po_vm["output"].as<string_t>();

        // load input file
        io::buffer_t data;
        if (!io::load_buffer(cmd_input, data))
        {
                log_error() << "failed to load input file from <" << cmd_input << ">!";
                return EXIT_FAILURE;
        }

        log_info () << "input: " << data.size() << " bytes.";

        const string_t name = text::lower(text::replace(boost::filesystem::basename(cmd_input), '-', '_'));
        const string_t tab(8, ' ');
        const size_t rowsize = 32;

        const string_t path_header = text::lower(text::replace(cmd_output, '-', '_')) + ".h";
        const string_t path_source = text::lower(text::replace(cmd_output, '-', '_')) + ".cpp";

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
        os_header << "namespace cortex\n";
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

        os_source << "#include " << boost::filesystem::path(path_header).filename() << "\n";
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

        os_source << "const char* cortex::get_" << name << "_data()\n";
        os_source << "{\n";
        os_source << tab << "return (const char*)data;\n";
        os_source << "}\n\n";

        os_source << "std::size_t cortex::get_" << name << "_size()\n";
        os_source << "{\n";
        os_source << tab << "return sizeof(data) / sizeof(unsigned char);\n";
        os_source << "}\n\n";

        os_source << "const char* cortex::get_" << name << "_name()\n";
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
        log_info() << cortex::done;
        return EXIT_SUCCESS;
}
