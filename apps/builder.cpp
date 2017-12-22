#include "io/io.h"
#include "model.h"
#include "logger.h"
#include "text/cmdline.h"
#include "layers/builder.h"

using namespace nano;

std::vector<tensor_size_t> parse_config(const string_t& str)
{
        std::vector<tensor_size_t> config;
        if (!str.empty() && str != ",")
        {
                for (const auto& token : split(str, ","))
                {
                        try
                        {
                                config.push_back(from_string<tensor_size_t>(token));
                        }
                        catch (std::exception&)
                        {
                                log_error() << "invalid token [" << token << "] in [" << str << "]!";
                                exit(EXIT_FAILURE);
                        }
                }
        }

        return config;
}

affine_node_configs_t affine_config(const string_t& str)
{
        const auto values = parse_config(str);
        if ((values.size() % 3) != 0)
        {
                log_error() << "invalid configuration for the affine nodes!";
                exit(EXIT_FAILURE);
        }

        affine_node_configs_t config;
        for (size_t i = 0; i < values.size(); i += 3)
        {
                config.push_back({values[i + 0], values[i + 1], values[i + 2]});
        }

        return config;
}

conv3d_node_configs_t conv3d_config(const string_t& str)
{
        const auto values = parse_config(str);
        if ((values.size() % 6) != 0)
        {
                log_error() << "invalid configuration for the convolution nodes!";
                exit(EXIT_FAILURE);
        }

        conv3d_node_configs_t config;
        for (size_t i = 0; i < values.size(); i += 6)
        {
                config.push_back({values[i + 0], values[i + 1], values[i + 2], values[i + 3], values[i + 4], values[i + 5]});
        }

        return config;
}

int main(int argc, const char *argv[])
{
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
        cmdline.add("", "cnn",          "construct a convolution network");
        cmdline.add("", "mlp",          "construct a multi-layer perceptron network");
        cmdline.add("", "linear",       "construct a linear model");
        cmdline.add("", "res-mlp",      "construct a residual MLP (multi-layer perceptron) network (*beta*)");
        cmdline.add("", "conv3d-param", "conv3d nodes like [omaps,krows,kcols,kconn,kdrow,kdcol,]+", ",");
        cmdline.add("", "affine-param", "affine nodes like [omaps,orows,ocols,]+", ",");
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

        if (    (!cmdline.has("cnn") && !cmdline.has("mlp") && !cmdline.has("linear") && !cmdline.has("res-mlp")) ||
                !cmdline.has("json"))
        {
                cmdline.usage();
        }

        // construct model
        model_t model;

        const auto conv3d_param = conv3d_config(cmdline.get<string_t>("conv3d-param"));
        const auto affine_param = affine_config(cmdline.get<string_t>("affine-param"));

        if (cmdline.has("linear"))
        {
                make_linear(model, cmd_omaps, cmd_orows, cmd_ocols, cmd_act_type);
        }
        else if(cmdline.has("mlp"))
        {
                make_mlp(model, affine_param, cmd_omaps, cmd_orows, cmd_ocols, cmd_act_type);
        }
        else if (cmdline.has("cnn"))
        {
                make_cnn(model, conv3d_param, affine_param, cmd_omaps, cmd_orows, cmd_ocols, cmd_act_type);
        }
        else if (cmdline.has("res-mlp"))
        {
                make_residual_mlp(model, affine_param, cmd_omaps, cmd_orows, cmd_ocols, cmd_act_type);
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

        log_info() << std::endl << writer.str();

        if (!nano::save_string(cmdline.get<string_t>("json"), writer.str()))
        {
                log_error() << "failed to open output file!";
                return EXIT_FAILURE;
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
