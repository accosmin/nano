#include "nano.h"
#include "text/cmdline.h"
#include "measure_and_log.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        const auto model_ids = nano::get_models().ids();

        // parse the command line
        nano::cmdline_t cmdline("describe a model");
        cmdline.add("", "model",                ("model to choose from: " + nano::concatenate(model_ids, ", ")).c_str());
        cmdline.add("", "model-file",           "filepath to load the model from");

        cmdline.process(argc, argv);

        if (!cmdline.has("model"))
        {
                cmdline.usage();
        }

        // check arguments and options
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_model_file = cmdline.get<string_t>("model-file");

        // create & load model
        const auto model = nano::get_models().get(cmd_model);

        nano::measure_critical_and_log(
                [&] () { return model->load(cmd_model_file); },
                "load model <" + cmd_model + ">");

        model->describe();

        // OK
        nano::log_info() << nano::done;
        return EXIT_SUCCESS;
}
