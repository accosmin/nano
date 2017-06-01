#include "loss.h"
#include "task.h"
#include "model.h"
#include "class.h"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "vision/color.h"
#include "tasks/charset.h"
#include "chrono/measure.h"
#include "tensor/numeric.h"
#include "measure_and_log.h"
#include "layers/make_layers.h"
#include "text/table_row_mark.h"
#include <iostream>

using namespace nano;

template <typename tunits>
auto measure_accumulator(accumulator_t& acc, const task_t& task, const fold_t& fold, const size_t trials = 1)
{
        const auto duration = measure<tunits>([&] () { acc.update(task, fold); }, trials);
        return duration.count();
}

void print_probes(table_t& table, const string_t& basename, const std::vector<probes_t>& probes, const probes_t& probes0)
{
        for (const auto& probe0 : probes0)
        {
                auto& row = table.append();
                row << (basename + probe0.fullname());

                for (const auto& probesx : probes)
                {
                        for (const auto& probex : probesx)
                        {
                                if (probe0.fullname() == probex.fullname())
                                {
                                        row << probe0.timings().avg();
                                }
                        }
                }
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

        const size_t cmd_min_nthreads = 1;
        const size_t cmd_max_nthreads = logical_cpus();

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
        table_t ftable; ftable.header() << "forward [us/sample]";
        table_t btable; btable.header() << "backward [us/sample]";

        for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
        {
                ftable.header() << (to_string(nthreads) + "xCPU");
                btable.header() << (to_string(nthreads) + "xCPU");
        }

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

                auto& frow = ftable.append() << (cmd_name + " (" + to_string(model->psize()) + ")");
                auto& brow = btable.append() << (cmd_name + " (" + to_string(model->psize()) + ")");

                const auto fold = fold_t{0, protocol::train};

                std::vector<probes_t> fprobes(cmd_max_nthreads + 1);
                std::vector<probes_t> bprobes(cmd_max_nthreads + 1);

                // process the samples
                for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
                {
                        if (cmd_forward)
                        {
                                accumulator_t acc(*model, *loss);
                                acc.threads(nthreads);
                                acc.mode(accumulator_t::type::value);

                                const auto usecs = measure_accumulator<microseconds_t>(acc, *task, fold);
                                const auto count = acc.vstats().count();

                                log_info() << "<<< processed [" << count << "] forward samples in " << usecs << " us.";

                                frow << idiv(usecs, count);
                                fprobes[nthreads] = acc.probes();
                        }

                        if (cmd_backward)
                        {
                                accumulator_t acc(*model, *loss);
                                acc.threads(nthreads);
                                acc.mode(accumulator_t::type::vgrad);

                                const auto usecs = measure_accumulator<microseconds_t>(acc, *task, fold);
                                const auto count = acc.vstats().count();

                                log_info() << "<<< processed [" << count << "] backward samples in " << usecs << " us.";

                                brow << idiv(usecs, count);
                                bprobes[nthreads] = acc.probes();
                        }
                }

                // detailed per-component (e.g. per-layer) timing information
                if (cmd_forward && cmd_detailed)
                {
                        print_probes(ftable, ">", fprobes, fprobes[cmd_min_nthreads]);
                }

                if (cmd_backward && cmd_detailed)
                {
                        print_probes(btable, ">", bprobes, bprobes[cmd_min_nthreads]);
                }
        }

        // print results
        if (cmd_forward)
        {
                ftable.mark(make_table_mark_minimum_percentage_cols<size_t>(5));
                std::cout << ftable;
        }
        if (cmd_backward)
        {
                btable.mark(make_table_mark_minimum_percentage_cols<size_t>(5));
                std::cout << btable;
        }

        // OK
        return EXIT_SUCCESS;
}
