#include "nano.h"
#include "class.h"
#include "measure.hpp"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "tensor/numeric.hpp"
#include "measure_and_log.hpp"
#include "layers/make_layers.h"
#include "tasks/task_charset.h"
#include "text/table_row_mark.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("benchmark models");
        cmdline.add("s", "samples",     "number of samples to use [100, 100000]", "10000");
        cmdline.add("c", "conn",        "plane connectivity for convolution networks [1, 16]", "8");
        cmdline.add("", "mlps",         "benchmark MLP models");
        cmdline.add("", "convnets",     "benchmark convolution networks");
        cmdline.add("", "forward",      "evaluate the \'forward\' pass (output)");
        cmdline.add("", "backward",     "evaluate the \'backward' pass (gradient)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_samples = nano::clamp(cmdline.get<size_t>("samples"), 100, 100 * 1000);
        const auto cmd_conn = nano::clamp(cmdline.get<int>("conn"), 1, 16);
        const auto cmd_forward = cmdline.has("forward");
        const auto cmd_backward = cmdline.has("backward");
        const auto cmd_mlps = cmdline.has("mlps");
        const auto cmd_convnets = cmdline.has("convnets");

        if (!cmd_forward && !cmd_backward)
        {
                cmdline.usage();
        }

        if (!cmd_mlps && !cmd_convnets)
        {
                cmdline.usage();
        }

        const tensor_size_t cmd_rows = 28;
        const tensor_size_t cmd_cols = 28;
        const color_mode cmd_color = color_mode::luma;

        const size_t cmd_min_nthreads = 1;
        const size_t cmd_max_nthreads = nano::logical_cpus();

        // generate synthetic task
        charset_task_t task(charset::digit, cmd_color, cmd_rows, cmd_cols, cmd_samples);
        task.load();

        // construct models
        const string_t mlp0;
        const string_t mlp1 = mlp0 + make_affine_layer(128);
        const string_t mlp2 = mlp1 + make_affine_layer(128);
        const string_t mlp3 = mlp2 + make_affine_layer(128);
        const string_t mlp4 = mlp3 + make_affine_layer(128);
        const string_t mlp5 = mlp4 + make_affine_layer(128);

        const string_t convnet0_k2d;
        const string_t convnet1_k2d = convnet0_k2d + make_conv_layer("conv-k2d", 64, 9, 9, 1);
        const string_t convnet2_k2d = convnet1_k2d + make_conv_layer("conv-k2d", 64, 7, 7, cmd_conn);
        const string_t convnet3_k2d = convnet2_k2d + make_conv_layer("conv-k2d", 64, 5, 5, cmd_conn);
        const string_t convnet4_k2d = convnet3_k2d + make_conv_layer("conv-k2d", 64, 5, 5, cmd_conn);
        const string_t convnet5_k2d = convnet4_k2d + make_conv_layer("conv-k2d", 64, 3, 3, cmd_conn);

        const string_t convnet1_toe = nano::replace(convnet1_k2d, "conv-k2d", "conv-toe");
        const string_t convnet2_toe = nano::replace(convnet2_k2d, "conv-k2d", "conv-toe");
        const string_t convnet3_toe = nano::replace(convnet3_k2d, "conv-k2d", "conv-toe");
        const string_t convnet4_toe = nano::replace(convnet4_k2d, "conv-k2d", "conv-toe");
        const string_t convnet5_toe = nano::replace(convnet5_k2d, "conv-k2d", "conv-toe");

        const string_t outlayer = make_output_layer(task.osize());

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
        }
        if (cmd_convnets)
        {
                DEFINE(convnet1_k2d);
                DEFINE(convnet1_toe);
                DEFINE(convnet2_k2d);
                DEFINE(convnet2_toe);
                DEFINE(convnet3_k2d);
                DEFINE(convnet3_toe);
                DEFINE(convnet4_k2d);
                DEFINE(convnet4_toe);
                DEFINE(convnet5_k2d);
                DEFINE(convnet5_toe);
        }

        #undef DEFINE

        const auto loss = nano::get_losses().get("logistic");
        const auto criterion = nano::get_criteria().get("avg");

        // construct tables to compare models
        nano::table_t ftable("model-forward [us] / 1000 samples");
        nano::table_t btable("model-backward [us] / 1000 samples");

        for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
        {
                ftable.header() << (nano::to_string(nthreads) + "xCPU");
                btable.header() << (nano::to_string(nthreads) + "xCPU");
        }

        // evaluate models
        for (const auto& config : networks)
        {
                const string_t cmd_network = config.first;
                const string_t cmd_name = config.second;

                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                const auto model = nano::get_models().get("forward-network", cmd_network);
                model->resize(task, true);
                model->random_params();

                nano::table_row_t& frow = ftable.append(cmd_name + " (" + nano::to_string(model->psize()) + ")");
                nano::table_row_t& brow = btable.append(cmd_name + " (" + nano::to_string(model->psize()) + ")");

                const auto fold = fold_t{0, protocol::train};

                // process the samples
                for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
                {
                        if (cmd_forward)
                        {
                                accumulator_t lacc(*model, *loss, *criterion, criterion_t::type::value, scalar_t(0.1));
                                lacc.set_threads(nthreads);

                                const auto duration = nano::measure_robustly_usec([&] ()
                                {
                                        lacc.reset();
                                        lacc.update(task, fold);
                                }, 1);

                                log_info() << "<<< processed [" << lacc.count()
                                           << "] forward samples in " << duration.count() << " us.";

                                frow << idiv(static_cast<size_t>(duration.count()) * 1000, lacc.count());
                        }

                        if (cmd_backward)
                        {
                                accumulator_t gacc(*model, *loss, *criterion, criterion_t::type::vgrad, scalar_t(0.1));
                                gacc.set_threads(nthreads);

                                const auto duration = nano::measure_robustly_usec([&] ()
                                {
                                        gacc.reset();
                                        gacc.update(task, fold);
                                }, 1);

                                log_info() << "<<< processed [" << gacc.count()
                                           << "] backward samples in " << duration.count() << " us.";

                                brow << idiv(static_cast<size_t>(duration.count()) * 1000, gacc.count());
                        }
                }

                log_info();
        }

        // print results
        if (cmd_forward)
        {
                ftable.mark(nano::make_table_mark_minimum_percentage_cols<size_t>(5));
                ftable.print(std::cout);
        }
        log_info();
        if (cmd_backward)
        {
                btable.mark(nano::make_table_mark_minimum_percentage_cols<size_t>(5));
                btable.print(std::cout);
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
