#include "nanocv.h"
#include "common/io_utar.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        using namespace ncv;

        ncv::init();
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "test program");
        po_desc.add_options()("input,i",
                boost::program_options::value<ncv::string_t>(),
                "input tar archive path (.tar, .gz, .bz2)");
	
        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("input") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_input = po_vm["input"].as<string_t>();

        // callback
        const auto callback = [](const string_t& filename, const std::vector<char>& data)
        {
                log_info() << "untar: callback(" << filename << ", " << data.size() << " bytes)";
        };

        // decode archive
        ncv::timer_t timer;
        if (!io::untar(cmd_input, callback, "untar: ", "untar: "))
        {
                return EXIT_FAILURE;
        }
        else
        {
                log_info() << "untar: >>> loaded in " << timer.elapsed() << ".";

                // OK
                log_info() << ncv::done;
                return EXIT_SUCCESS;
        }
}
