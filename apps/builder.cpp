#include "model.h"
#include "logger.h"
#include "text/cmdline.h"
#include "layers/builder.h"
#include <fstream>

int main(int argc, const char *argv[])
{
        using namespace nano;

        strings_t activations;
        for (const auto& layer_id : get_layers().ids())
        {
                if (is_activation_node(layer_id))
                {
                        activations.push_back(layer_id);
                }
        }

        // parse the command line
        cmdline_t cmdline("construct and export models using predefined architectures");
        cmdline.add("", "mlp",          "construct a MLP (multi-layer perceptron) network");
        cmdline.add("", "mlp-params",   "number of feature maps per affine layer (e.g. 128,256,128)");
        cmdline.add("", "act-type",     "activation type " + join(activations), "act-snorm");
        cmdline.add("", "imaps",        "number of input feature maps", 3);
        cmdline.add("", "irows",        "number of input rows", 32);
        cmdline.add("", "icols",        "number of input cols", 32);
        cmdline.add("", "omaps",        "number of output feature maps", 10);
        cmdline.add("", "orows",        "number of output rows", 1);
        cmdline.add("", "ocols",        "number of output cols", 1);
        cmdline.add("", "json",         "path where to save the model description (.json)");

        cmdline.process(argc, argv);

        const auto cmd_act_type = cmdline.get<string_t>("act-type");
        const auto cmd_imaps = cmdline.get<tensor_size_t>("imaps");
        const auto cmd_irows = cmdline.get<tensor_size_t>("irows");
        const auto cmd_icols = cmdline.get<tensor_size_t>("icols");
        const auto cmd_omaps = cmdline.get<tensor_size_t>("omaps");
        const auto cmd_orows = cmdline.get<tensor_size_t>("orows");
        const auto cmd_ocols = cmdline.get<tensor_size_t>("ocols");

        if (    !cmdline.has("mlp") ||
                !cmdline.has("json"))
        {
                cmdline.usage();
        }

        // construct model
        model_t model;

        if (cmdline.has("mlp"))
        {
                std::vector<tensor_size_t> cmd_mlp_params;
                for (const auto& token : nano::split(cmdline.get<string_t>("mlp-params"), ","))
                {
                        cmd_mlp_params.push_back(from_string<tensor_size_t>(token));
                }

                make_mlp(model, cmd_mlp_params, cmd_omaps, cmd_orows, cmd_ocols, cmd_act_type);
        }

        // check model
        if (    !model.done() ||
                !model.resize(make_dims(cmd_imaps, cmd_irows, cmd_icols), make_dims(cmd_omaps, cmd_orows, cmd_ocols)))
        {
                log_error() << "invalid configuration!";
                return EXIT_FAILURE;
        }
        model.describe();

        // save model description to file
        json_writer_t writer;
        model.config(writer);

        std::ofstream out(cmdline.get<string_t>("json"));
        if (!out.is_open())
        {
                log_error() << "failed to open output file!";
                return EXIT_FAILURE;
        }

        out << writer.str();
        out.close();

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
