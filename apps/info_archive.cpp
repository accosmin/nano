#include "file/archive.h"
#include "cortex/string.h"
#include "cortex/util/timer.h"
#include "cortex/util/logger.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        using namespace cortex;
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "display the structure of the given archive");
        po_desc.add_options()("input,i",
                boost::program_options::value<cortex::string_t>(),
                "input archive path (.tar, .gz, .bz2, .tar.gz, .tar.bz2)");
	
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
        const auto callback = [] (const string_t& filename, const std::vector<char>& data)
        {
                log_info() << "decode: callback(" << filename << ", " << data.size() << " bytes)";
                return true;
        };
        const auto error_callback = [] (const string_t& message)
        {
                log_error() << "decode: " << message;
        };

        // decode archive
        cortex::timer_t timer;
        if (!file::unarchive(cmd_input, callback, error_callback))
        {
                return EXIT_FAILURE;
        }
        else
        {
                log_info() << "<" << cmd_input << "> loaded in " << timer.elapsed() << ".";

                // OK
                log_info() << cortex::done;
                return EXIT_SUCCESS;
        }
}
