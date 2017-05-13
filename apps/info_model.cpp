#include "model.h"
#include "text/cmdline.h"
#include "measure_and_log.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("describe a model");
        cmdline.add("", "model",        "[" + concatenate(get_models().ids()) + "]");
        cmdline.add("", "model-file",   "filepath to load the model from");

        cmdline.process(argc, argv);

        if (!cmdline.has("model"))
        {
                cmdline.usage();
        }

        // check arguments and options
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_model_file = cmdline.get<string_t>("model-file");

        // create & load model
        const auto model = get_models().get(cmd_model);

        measure_critical_and_log(
                [&] () { return model->load(cmd_model_file); },
                "load model <" + cmd_model + ">");

        model->describe();

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
