#include "loss.h"
#include "task.h"
#include "model.h"
#include "logger.h"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "vision/color.h"
#include "tasks/charset.h"
#include "tensor/numeric.h"
#include "text/algorithm.h"
#include "layers/make_layers.h"
#include <iostream>

using namespace nano;

void append(table_t& table, const string_t& name, const probes_t& probes, const bool detailed)
{
        for (const auto& probe : probes)
        {
                if (!starts_with(probe.fullname(), "network") && !detailed)
                {
                        continue;
                }

                auto& row = table.append();
                row << (name + " " + probe.fullname());
                row << probe.flops();
                if (probe.timings().min() < int64_t(1))
                {
                        row << "-";
                }
                else
                {
                        row << probe.gflops();
                }
                row << probe.timings().min() << probe.timings().avg() << probe.timings().max();
        }
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark models");
        cmdline.add("s", "samples",     "number of samples to use [100, 10000]", "1000");
        cmdline.add("c", "conn",        "plane connectivity for convolution networks [1, 16]", "1");
        cmdline.add("", "mlps",         "benchmark MLP models");
        cmdline.add("", "convnets",     "benchmark convolution networks");
        cmdline.add("", "forward",      "evaluate the \'forward\' pass (output)");
        cmdline.add("", "backward",     "evaluate the \'backward' pass (gradient)");
        cmdline.add("", "activation",   "activation layer", "act-snorm");
        cmdline.add("", "detailed",     "print detailed measurements (e.g. per-layer)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_samples = clamp(cmdline.get<size_t>("samples"), 100, 100 * 1000);
        const auto conn = clamp(cmdline.get<int>("conn"), 1, 16);
        const auto cmd_forward = cmdline.has("forward");
        const auto cmd_backward = cmdline.has("backward");
        const auto cmd_mlps = cmdline.has("mlps");
        const auto cmd_convnets = cmdline.has("convnets");
        const auto activation = cmdline.get("activation");
        const auto cmd_detailed = cmdline.has("detailed");

        if (!cmd_forward && !cmd_backward)
        {
                cmdline.usage();
        }

        if (!cmd_mlps && !cmd_convnets)
        {
                cmdline.usage();
        }

        const auto cmd_rows = 28;
        const auto cmd_cols = 28;
        const auto cmd_color = color_mode::luma;

        // generate synthetic task
        auto task = get_tasks().get("synth-charset", to_params(
                "type", charset_type::digit, "color", cmd_color, "irows", cmd_rows, "icols", cmd_cols, "count", cmd_samples));
        task->load();

        // construct models
        const string_t mlp0;
        const string_t mlp1 = mlp0 + make_affine_layer(256, activation);
        const string_t mlp2 = mlp1 + make_affine_layer(256, activation);
        const string_t mlp3 = mlp2 + make_affine_layer(256, activation);
        const string_t mlp4 = mlp3 + make_affine_layer(256, activation);
        const string_t mlp5 = mlp4 + make_affine_layer(256, activation);
        const string_t mlp6 = mlp5 + make_affine_layer(256, activation);
        const string_t mlp7 = mlp6 + make_affine_layer(128, activation);
        const string_t mlp8 = mlp7 + make_affine_layer(128, activation);
        const string_t mlp9 = mlp8 + make_affine_layer(128, activation);

        const string_t convnet0;
        const string_t convnet1 = convnet0 + make_conv3d_layer(32,  7, 7, 1, activation);
        const string_t convnet2 = convnet1 + make_conv3d_layer(32,  7, 7, conn, activation);
        const string_t convnet3 = convnet2 + make_conv3d_layer(64,  5, 5, conn, activation);
        const string_t convnet4 = convnet3 + make_conv3d_layer(64,  5, 5, conn, activation);
        const string_t convnet5 = convnet4 + make_conv3d_layer(128, 3, 3, conn, activation);
        const string_t convnet6 = convnet5 + make_conv3d_layer(128, 3, 3, conn, activation);
        const string_t convnet7 = convnet6 + make_conv3d_layer(128, 3, 3, conn, activation);
        const string_t convnet8 = convnet7 + make_conv3d_layer(256, 1, 1, conn, activation);
        const string_t convnet9 = convnet8 + make_conv3d_layer(256, 1, 1, conn, activation);

        const string_t outlayer = make_output_layer(task->odims());

        std::vector<std::pair<string_t, string_t>> networks;
        if (cmd_mlps)
        {
                networks.emplace_back(mlp0 + outlayer, "mlp0");
                networks.emplace_back(mlp1 + outlayer, "mlp1");
                networks.emplace_back(mlp2 + outlayer, "mlp2");
                networks.emplace_back(mlp3 + outlayer, "mlp3");
                networks.emplace_back(mlp4 + outlayer, "mlp4");
                networks.emplace_back(mlp5 + outlayer, "mlp5");
                networks.emplace_back(mlp6 + outlayer, "mlp6");
                networks.emplace_back(mlp7 + outlayer, "mlp7");
                networks.emplace_back(mlp8 + outlayer, "mlp8");
                networks.emplace_back(mlp9 + outlayer, "mlp9");
        }
        if (cmd_convnets)
        {
                networks.emplace_back(convnet1 + outlayer, "convnet1");
                networks.emplace_back(convnet2 + outlayer, "convnet2");
                networks.emplace_back(convnet3 + outlayer, "convnet3");
                networks.emplace_back(convnet4 + outlayer, "convnet4");
                networks.emplace_back(convnet5 + outlayer, "convnet5");
                networks.emplace_back(convnet6 + outlayer, "convnet6");
                networks.emplace_back(convnet7 + outlayer, "convnet7");
                networks.emplace_back(convnet8 + outlayer, "convnet8");
                networks.emplace_back(convnet9 + outlayer, "convnet9");
        }

        const auto loss = get_losses().get("s-logistic");

        // construct tables to compare models
        table_t table;
        table.header() << "network" << "#flops" << "gflop/s" << "min[us]" << "avg[us]" << "max[us]";
        table.delim();

        // evaluate models
        for (const auto& config : networks)
        {
                const auto cmd_network = config.first;
                const auto cmd_name = config.second;

                // create feed-forward network
                const auto model = get_models().get("forward-network", cmd_network);
                model->configure(*task);
                model->random();
                model->describe();

                const auto name = cmd_name + " (" + to_string(model->psize()) + ")";
                const auto fold = fold_t{0, protocol::train};
                const auto size = task->size(fold);

                // measure processing
                if (cmd_forward && !cmd_backward)
                {
                        accumulator_t acc(*model, *loss);
                        acc.threads(1);
                        acc.mode(accumulator_t::type::value);
                        acc.update(*task, fold);

                        log_info() << "<<< processed [" << size << "] forward samples for " << name << ".";

                        append(table, name, acc.probes(), cmd_detailed);
                }
                else
                {
                        accumulator_t acc(*model, *loss);
                        acc.threads(1);
                        acc.mode(accumulator_t::type::vgrad);
                        acc.update(*task, fold);

                        log_info() << "<<< processed [" << size << "] backward samples for " << name << ".";

                        append(table, name, acc.probes(), cmd_detailed);
                }

                const auto last = config == *networks.rbegin();
                if (cmd_detailed && !last)
                {
                        table.delim();
                }
        }

        // print results
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
