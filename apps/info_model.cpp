#include "model.h"
#include "text/cmdline.h"
#include "measure_and_log.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("describe a model");
        cmdline.add("", "model-file",   "filepath to load the model from");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_model_file = cmdline.get<string_t>("model-file");

        // create & load model
        model_t model;
        measure_critical_and_log(
                [&] () { return model.load(cmd_model_file); },
                "load model");
        model.describe();

        // todo: print L2 stats per layer & per processing unit (aka neuron)
        // todo: print correlation factors per layer (aka how correlated the processing units are per layer)

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
