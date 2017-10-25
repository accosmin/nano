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

void append(table_t& table, const string_t& name, const tensor_size_t params, const size_t minibatch,
        const probes_t& probes, const bool detailed)
{
        for (const auto& probe : probes)
        {
                if (!starts_with(probe.fullname(), "network") && !detailed)
                {
                        continue;
                }

                auto& row = table.append();
                row << (name + " " + probe.fullname()) << params << probe.flops() << minibatch;
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
        cmdline.add("", "min-count",    "minimum number of samples in minibatch [1, 16]",  "1");
        cmdline.add("", "max-count",    "maximum number of samples in minibatch [1, 128]", "16");

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
        const auto cmd_min_count = clamp(cmdline.get<size_t>("min-count"), 1, 16);
        const auto cmd_max_count = clamp(cmdline.get<size_t>("max-count"), cmd_min_count, 128);

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
        const string_t mlp0 = "normalize:type=plane;";
        const string_t mlp1 = mlp0 + make_affine_layer(1024, 1, 1, activation);
        const string_t mlp2 = mlp1 + make_affine_layer(1024, 1, 1, activation);
        const string_t mlp3 = mlp2 + make_affine_layer(1024, 1, 1, activation);
        const string_t mlp4 = mlp3 + make_affine_layer(1024, 1, 1, activation);
        const string_t mlp5 = mlp3 + make_affine_layer(1024, 1, 1, activation);
        const string_t mlp6 = mlp3 + make_affine_layer(1024, 1, 1, activation);

        const string_t convnet0 = "normalize:type=plane;";
        const string_t convnet1 = convnet0 + make_conv3d_layer(128, 7, 7, 1, activation);
        const string_t convnet2 = convnet1 + make_conv3d_layer(128, 7, 7, conn, activation);
        const string_t convnet3 = convnet2 + make_conv3d_layer(128, 5, 5, conn, activation);
        const string_t convnet4 = convnet3 + make_conv3d_layer(128, 5, 5, conn, activation);
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
        table.header() << "network" << "#params" << "#flops" << "batch" << "gflop/s" << "min[us]" << "avg[us]" << "max[us]";
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

                const auto fold = fold_t{0, protocol::train};
                const auto size = task->size(fold);

                for (size_t count = cmd_min_count; count <= cmd_max_count; count *= 2)
                {
                        // measure processing
                        accumulator_t acc(*model, *loss);
                        acc.threads(1);
                        acc.mode((cmd_forward && !cmd_backward) ? accumulator_t::type::value : accumulator_t::type::vgrad);

                        for (size_t i = 0; i + count < size; i += count)
                        {
                                acc.update(*task, fold, i, i + count);
                        }

                        log_info()
                                << "<<< processed [" << size << "] samples using "
                                << cmd_name << " and minibatch of " << count << ".";

                        append(table, cmd_name, model->psize(), count, acc.probes(), cmd_detailed);

                        if (cmd_detailed && count * 2 <= cmd_max_count)
                        {
                                table.delim();
                        }
                }

                const auto last = config == *networks.rbegin();
                if (!cmd_detailed && !last)
                {
                        table.delim();
                }
        }

        // print results
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
