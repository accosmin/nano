#include "nanocv.h"
#include "file/archive.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        using namespace ncv;

        ncv::init();
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "display the structure of the given archive");
        po_desc.add_options()("input,i",
                boost::program_options::value<ncv::string_t>(),
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
        const auto callback = [](const string_t& filename, const std::vector<char>& data)
        {
                log_info() << "decode: callback(" << filename << ", " << data.size() << " bytes)";
                return true;
        };

        // decode archive
        ncv::timer_t timer;
        if (!io::decode(cmd_input, "decode: ", callback))
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
