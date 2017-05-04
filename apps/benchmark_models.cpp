#include "nano.h"
#include "class.h"
#include "measure.h"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "vision/color.h"
#include "tasks/charset.h"
#include "tensor/numeric.h"
#include "measure_and_log.h"
#include "layers/make_layers.h"
#include "text/table_row_mark.h"
#include <iostream>

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("benchmark models");
        cmdline.add("s", "samples",     "number of samples to use [100, 10000]", "1000");
        cmdline.add("c", "conn",        "plane connectivity for convolution networks [1, 16]", "8");
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
        const string_t mlp1 = mlp0 + make_affine_layer(128, activation);
        const string_t mlp2 = mlp1 + make_affine_layer(512, activation);
        const string_t mlp3 = mlp2 + make_affine_layer(128, activation);
        const string_t mlp4 = mlp3 + make_affine_layer(512, activation);
        const string_t mlp5 = mlp4 + make_affine_layer(128, activation);
        const string_t mlp6 = mlp5 + make_affine_layer(512, activation);
        const string_t mlp7 = mlp6 + make_affine_layer(128, activation);

        const string_t convnet0;
        const string_t convnet1 = convnet0 + make_conv_layer(128, 7, 7, 1, activation);
        const string_t convnet2 = convnet1 + make_conv_layer(128, 7, 7, conn, activation);
        const string_t convnet3 = convnet2 + make_conv_layer(128, 5, 5, conn, activation);
        const string_t convnet4 = convnet3 + make_conv_layer(128, 5, 5, conn, activation);
        const string_t convnet5 = convnet4 + make_conv_layer(128, 3, 3, conn, activation);
        const string_t convnet6 = convnet5 + make_conv_layer(128, 3, 3, conn, activation);
        const string_t convnet7 = convnet6 + make_conv_layer(128, 3, 3, conn, activation);

        const string_t outlayer = make_output_layer(task->odims());

        std::vector<std::pair<string_t, string_t>> networks;
        #define DEFINE(config) networks.emplace_back(config + outlayer, NANO_STRINGIFY(config))
        if (cmd_mlps)
        {
                DEFINE(mlp0);
                DEFINE(mlp1);
                DEFINE(mlp2);
                DEFINE(mlp3);
                DEFINE(mlp4);
                DEFINE(mlp5);
                DEFINE(mlp6);
                DEFINE(mlp7);
        }
        if (cmd_convnets)
        {
                DEFINE(convnet1);
                DEFINE(convnet2);
                DEFINE(convnet3);
                DEFINE(convnet4);
                DEFINE(convnet5);
                DEFINE(convnet6);
                DEFINE(convnet7);
        }
        #undef DEFINE

        const auto loss = get_losses().get("logistic");
        const auto criterion = get_criteria().get("avg");

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
                const string_t cmd_network = config.first;
                const string_t cmd_name = config.second;

                // create feed-forward network
                const auto model = get_models().get("forward-network", cmd_network);
                model->configure(*task);
                model->random();
                model->describe();

                auto& frow = ftable.append() << (cmd_name + " (" + to_string(model->psize()) + ")");
                auto& brow = btable.append() << (cmd_name + " (" + to_string(model->psize()) + ")");

                const auto fold = fold_t{0, protocol::train};

                std::vector<timings_t> ftimings(cmd_max_nthreads + 1);
                std::vector<timings_t> btimings(cmd_max_nthreads + 1);

                // process the samples
                for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
                {
                        accumulator_t acc(*model, *loss, *criterion);
                        acc.lambda(scalar_t(0.1));
                        acc.threads(nthreads);

                        if (cmd_forward)
                        {
                                const auto duration = measure_robustly<microseconds_t>([&] ()
                                {
                                        acc.mode(criterion_t::type::value);
                                        acc.update(*task, fold);
                                }, 1);

                                log_info() << "<<< processed [" << acc.count()
                                           << "] forward samples in " << duration.count() << " us.";

                                frow << idiv(static_cast<size_t>(duration.count()), acc.count());

                                ftimings[nthreads] = acc.timings();
                        }

                        if (cmd_backward)
                        {
                                const auto duration = measure_robustly<microseconds_t>([&] ()
                                {
                                        acc.mode(criterion_t::type::vgrad);
                                        acc.update(*task, fold);
                                }, 1);

                                log_info() << "<<< processed [" << acc.count()
                                           << "] backward samples in " << duration.count() << " us.";

                                brow << idiv(static_cast<size_t>(duration.count()), acc.count());

                                btimings[nthreads] = acc.timings();
                        }
                }

                // detailed per-component (e.g. per-layer) timing information
                const auto print_timings = [&] (table_t& table, const string_t& basename,
                        const std::vector<timings_t>& timings)
                {
                        for (const auto& timing0 : timings[cmd_min_nthreads])
                        {
                                auto& row = table.append();
                                row << (basename + timing0.first);

                                for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
                                {
                                        const auto& timingT = timings[nthreads];
                                        assert(timingT.find(timing0.first) != timingT.end());
                                        row << timingT.find(timing0.first)->second.avg();
                                }
                        }
                };

                if (cmd_forward && cmd_detailed)
                {
                        print_timings(ftable, ">", ftimings);
                }

                if (cmd_backward && cmd_detailed)
                {
                        print_timings(btable, ">", btimings);
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
