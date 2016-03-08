#include "stringi.h"
#include "io/archive.h"
#include "text/cmdline.h"
#include "cortex/util/timer.h"
#include "cortex/util/logger.h"

int main(int argc, char *argv[])
{
        using namespace cortex;
        
        // parse the command line
        text::cmdline_t cmdline("display the structure of the given archive");
        cmdline.add("i", "input",       "input archive path (.tar, .gz, .bz2, .tar.gz, .tar.bz2)");
	
        cmdline.process(argc, argv);
        		
        // check arguments and options
        const string_t cmd_input = cmdline.get("input");

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
        if (!io::unarchive(cmd_input, callback, error_callback))
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
