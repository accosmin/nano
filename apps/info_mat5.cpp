#include "timer.h"
#include "logger.h"
#include "stringi.h"
#include "io/mat5.h"
#include "text/cmdline.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("display the structure of the given matlab5 file");
        cmdline.add("i", "input",       "input matlab5 path (.mat)");

        cmdline.process(argc, argv);

        // check arguments and options
        const string_t cmd_input = cmdline.get("input");

        // callbacks
        const auto hcallback = [] (const mat5_header_t& header)
        {
                log_info() << "decode: [" << header.description() << "]";
                return true;
        };
        const auto scallback = [] (const mat5_section_t& section, istream_t& stream)
        {
                log_info() << "decode: " << section;
                return section.skip(stream);
        };
        const auto ecallback = [] (const string_t& message)
        {
                log_error() << "decode: " << message;
        };

        // load file
        nano::timer_t timer;
        if (!nano::load_mat5(cmd_input, hcallback, scallback, ecallback))
        {
                log_error() << "failed to load <" << cmd_input << ">!";
                return EXIT_FAILURE;
        }
        else
        {
                log_info() << "<" << cmd_input << "> loaded in " << timer.elapsed() << ".";
                log_info() << nano::done;
                return EXIT_SUCCESS;
        }
}
