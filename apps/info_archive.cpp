#include "stringi.h"
#include "core/archive.h"
#include "core/cmdline.h"
#include "core/checkpoint.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("display the structure of the given archive");
        cmdline.add("i", "input",       "input archive path (.tar, .gz, .bz2, .tar.gz, .tar.bz2)");

        cmdline.process(argc, argv);

        // check arguments and options
        const string_t cmd_input = cmdline.get("input");

        // callbacks
        const auto callback = [] (const string_t& filename, istream_t& stream)
        {
                log_info() << "decode: callback(" << filename << ", " << stream.skip() << " bytes)";
                return true;
        };
        const auto error_callback = [] (const string_t& message)
        {
                log_error() << "decode: " << message;
        };

        // load file
        checkpoint_t checkpoint;
        checkpoint.step(strcat("load archive from <", cmd_input, ">"));
        checkpoint.measure(nano::load_archive(cmd_input, callback, error_callback));

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
