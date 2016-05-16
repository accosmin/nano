#include "text/table.h"
#include "text/cmdline.h"
#include "cortex/class.h"
#include "cortex/cortex.h"
#include "math/clamp.hpp"
#include "thread/thread.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "tensor/numeric.hpp"
#include "cortex/measure.hpp"
#include "cortex/accumulator.h"
#include "text/table_row_mark.h"
#include "cortex/measure_and_log.hpp"
#include "cortex/layers/make_layers.h"
#include "cortex/tasks/task_charset.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("benchmark models");
        cmdline.add("s", "samples",     "number of samples to use [100, 100000]", "10000");
        cmdline.add("", "mlp",          "benchmark MLP models");
        cmdline.add("", "convnet",      "benchmark convolution networks");
        cmdline.add("", "forward",      "evaluate the \'forward\' pass (output)");
        cmdline.add("", "backward",     "evaluate the \'backward' pass (gradient)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_samples = nano::clamp(cmdline.get<size_t>("samples"), 100, 100 * 1000);
        const auto cmd_forward = cmdline.has("forward");
        const auto cmd_backward = cmdline.has("backward");
        const auto cmd_mlp = cmdline.has("mlp");
        const auto cmd_convnet = cmdline.has("convnet");

        if (!cmd_forward && !cmd_backward)
        {
                cmdline.usage();
        }

        if (!cmd_mlp && !cmd_convnet)
        {
                cmdline.usage();
        }

        const tensor_size_t cmd_rows = 28;
        const tensor_size_t cmd_cols = 28;
        const color_mode cmd_color = color_mode::luma;

        const size_t cmd_min_nthreads = 1;
        const size_t cmd_max_nthreads = thread::concurrency();

        // generate synthetic task
        charset_task_t task(charset::digit, cmd_color, cmd_rows, cmd_cols, cmd_samples);
        task.load();

        // construct models
        const string_t mlp0;
        const string_t mlp1 = mlp0 + make_affine_layer(100);
        const string_t mlp2 = mlp1 + make_affine_layer(100);
        const string_t mlp3 = mlp2 + make_affine_layer(100);
        const string_t mlp4 = mlp3 + make_affine_layer(100);
        const string_t mlp5 = mlp4 + make_affine_layer(100);

        const string_t convnet_9x9p_5x5p_3x3 =
                make_conv_pool_layer(16, 9, 9, 1) +
                make_conv_pool_layer(32, 5, 5, 2) +
                make_conv_layer(64, 3, 3, 4);

        const string_t convnet_7x7p_5x5p_3x3 =
                make_conv_pool_layer(16, 7, 7, 1) +
                make_conv_pool_layer(32, 5, 5, 2) +
                make_conv_layer(64, 3, 3, 4);

        const string_t convnet_11x11_9x9_7x7_3x3 =
                make_conv_layer(16, 11, 11, 1) +
                make_conv_layer(32, 9, 9, 2) +
                make_conv_layer(64, 7, 7, 4) +
                make_conv_layer(64, 3, 3, 8);

        const string_t convnet_11x11_9x9_5x5_5x5 =
                make_conv_layer(16, 11, 11, 1) +
                make_conv_layer(32, 9, 9, 2) +
                make_conv_layer(64, 5, 5, 4) +
                make_conv_layer(64, 5, 5, 8);

        const string_t convnet_9x9_7x7_7x7_5x5_3x3 =
                make_conv_layer(16, 9, 9, 1) +
                make_conv_layer(32, 7, 7, 2) +
                make_conv_layer(32, 7, 7, 4) +
                make_conv_layer(64, 5, 5, 4) +
                make_conv_layer(64, 3, 3, 8);

        const string_t convnet_7x7_7x7_5x5_5x5_5x5_3x3 =
                make_conv_layer(16, 7, 7, 1) +
                make_conv_layer(16, 7, 7, 2) +
                make_conv_layer(32, 5, 5, 2) +
                make_conv_layer(32, 5, 5, 4) +
                make_conv_layer(32, 5, 5, 4) +
                make_conv_layer(64, 3, 3, 4);

        const string_t outlayer = make_output_layer(task.osize());

        std::vector<std::pair<string_t, string_t>> networks;
        #define DEFINE(config) networks.emplace_back(config + outlayer, NANO_STRINGIFY(config))

        if (cmd_mlp)
        {
                DEFINE(mlp0);
                DEFINE(mlp1);
                DEFINE(mlp2);
                DEFINE(mlp3);
                DEFINE(mlp4);
                DEFINE(mlp5);
        }
        if (cmd_convnet)
        {
                DEFINE(convnet_9x9p_5x5p_3x3);
                DEFINE(convnet_7x7p_5x5p_3x3);
                DEFINE(convnet_11x11_9x9_7x7_3x3);
                DEFINE(convnet_11x11_9x9_5x5_5x5);
                DEFINE(convnet_9x9_7x7_7x7_5x5_3x3);
                DEFINE(convnet_7x7_7x7_5x5_5x5_5x5_3x3);
        }

        #undef DEFINE

        const auto loss = nano::get_losses().get("logistic");
        const auto criterion = nano::get_criteria().get("l2n-reg");

        // construct tables to compare models
        nano::table_t ftable("model-forward [ms] / 1000 samples");
        nano::table_t btable("model-backward [ms] / 1000 samples");

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
                                accumulator_t lacc(*model, *loss, *criterion, criterion_t::type::value, 0.1);
                                lacc.set_threads(nthreads);

                                const auto milis = nano::measure_robustly_msec([&] ()
                                {
                                        lacc.reset();
                                        lacc.update(task, fold);
                                }, 1);

                                log_info() << "<<< processed [" << lacc.count()
                                           << "] forward samples in " << milis.count() << " ms.";

                                frow << idiv(static_cast<size_t>(milis.count()) * 1000, lacc.count());
                        }

                        if (cmd_backward)
                        {
                                accumulator_t gacc(*model, *loss, *criterion, criterion_t::type::vgrad, 0.1);
                                gacc.set_threads(nthreads);

                                const auto milis = nano::measure_robustly_msec([&] ()
                                {
                                        gacc.reset();
                                        gacc.update(task, fold);
                                }, 1);

                                log_info() << "<<< processed [" << gacc.count()
                                           << "] backward samples in " << milis.count() << " ms.";

                                brow << idiv(static_cast<size_t>(milis.count()) * 1000, gacc.count());
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
