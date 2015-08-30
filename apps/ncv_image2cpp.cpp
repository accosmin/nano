#include "nanocv/string.h"
#include "nanocv/logger.h"
#include "nanocv/vision/image.h"
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        using namespace ncv;
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "test program");
        po_desc.add_options()("input,i",
                boost::program_options::value<ncv::string_t>(),
                "input image path");
        po_desc.add_options()("luma",
                "load the image as luma (grayscale)");
        po_desc.add_options()("rgba",
                "load the image as RGBA (color)");
        po_desc.add_options()("output,o",
                boost::program_options::value<ncv::string_t>(),
                "output base file name (to generate .h & .cpp)");
	
        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("input") ||
                !po_vm.count("output") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_input = po_vm["input"].as<string_t>();
        const string_t cmd_output = po_vm["output"].as<string_t>();

        const bool cmd_luma = po_vm.count("luma");
        const bool cmd_rgba = po_vm.count("rgba");        

        if (    (!cmd_luma && !cmd_rgba) ||
                (cmd_luma && cmd_rgba))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        // load input image
        image_t image;
        if (!(cmd_luma ? image.load_luma(cmd_input) : image.load_rgba(cmd_input)))
        {
                log_error() << "failed to load image from <" << cmd_input << ">!";
                return EXIT_FAILURE;
        }

        log_info () << "image: " << image.cols() << "x" << image.rows() << " pixels, "
                    << (image.is_luma() ? "[luma]" : "[rgba]") << ".";

        const string_t funcname = "get_" + boost::filesystem::basename(cmd_input);
        const string_t retname = (cmd_luma ? "luma_matrix_t" : "rgba_matrix_t");
        const string_t tab(8, ' ');

        const string_t path_header = cmd_output + ".h";
        const string_t path_source = cmd_output + ".cpp";

        // generate header
        std::ofstream os_header(path_header.c_str(), std::ios::out);
        if (!os_header.is_open())
        {
                log_error() << "failed to open header <" << path_header << ">!";
                return EXIT_FAILURE;
        }

        os_header << "#pragma once\n";
        os_header << "\n";
        os_header << "#include \"nanocv/vision/color.h\"\n";
        os_header << "\n";
        os_header << "namespace ncv\n";
        os_header << "{\n";
        os_header << tab << retname << " " << funcname << "();\n";
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
        os_source << "ncv::" << retname << " ncv::" << funcname << "()\n";
        os_source << "{\n";
        {
                const auto op = [&] (const auto& buff)
                {
                        os_source << tab << retname << " data(" << image.rows() << ", " << image.cols() << ");\n";
                        os_source << "\n";
                        for (coord_t r = 0; r < image.rows(); r ++)
                        {
                                os_source << tab;
                                for (coord_t c = 0; c < image.cols(); c ++)
                                {
                                        os_source << "data(" << r << ", " << c << ") = " << std::to_string(buff(r, c))
                                                  << ((c + 1 == image.cols()) ? "" : ", ");
                                }
                                os_source << ";\n\n";
                        }
                        os_source << "return data;\n";
                };

                if (cmd_luma)
                {
                        op(image.luma());
                }
                else
                {
                        op(image.rgba());
                }
        }
        os_source << "}\n";
        os_source << "\n";

        if (!os_source.good())
        {
                log_error() << "failed to write source <" << path_source << ">!";
                return EXIT_FAILURE;
        }
        os_source.close();
		
        // OK
        log_info() << ncv::done;
        return EXIT_SUCCESS;
}
