#include "nano.h"
#include "logger.h"
#include "measure.h"
#include "text/table.h"
#include "batch/types.h"
#include "text/cmdline.h"
#include "text/to_params.h"
#include "text/concatenate.h"
#include "tasks/task_charset.h"
#include "layers/make_layers.h"
#include <iostream>

using namespace nano;

template <typename tvalue>
static string_t stats_to_string(const stats_t<tvalue>& stats)
{
        return  to_string(static_cast<tvalue>(stats.avg()))
                + "+/-" + to_string(static_cast<tvalue>(stats.stdev()))
                + " [" + to_string(stats.min())
                + ", " + to_string(stats.max())
                + "]";
}

template <typename ttrainer>
static void evaluate_trainer(model_t& model, const string_t& name, const string_t& basepath,
        table_t& table, const vectors_t& x0s, const ttrainer& trainer)
{
        stats_t<scalar_t> values;
        stats_t<scalar_t> errors;
        stats_t<scalar_t> speeds;
        stats_t<scalar_t> timings;

        log_info() << "<<< running " << name << " ...";

        for (size_t i = 0; i < x0s.size(); ++ i)
        {
                const nano::timer_t timer;

                model.load_params(x0s[i]);

                const auto result = trainer();
                const auto opt_state = result.optimum_state();
                const auto opt_speed = convergence_speed(result.optimum_states());

                values(opt_state.m_test.m_value);
                errors(opt_state.m_test.m_error_avg);
                speeds(opt_speed);
                timings(static_cast<scalar_t>(timer.seconds().count()));

                const auto path = basepath + "-trial" + to_string(i) + ".state";

                const auto opt_states = result.optimum_states();
                save(path, opt_states);
        }

        table.append(name)
                << stats_to_string(values)
                << stats_to_string(errors)
                << stats_to_string(speeds)
                << stats_to_string(timings);
}

static void evaluate(model_t& model,
        const task_t& task, const size_t fold,
        const loss_t& loss, const criterion_t& criterion, const vectors_t& x0s, const size_t epochs,
        const strings_t& batch_optimizers,
        const strings_t& stoch_optimizers,
        const string_t& basename, const string_t& basepath, table_t& table)
{
        const auto nthreads = nano::logical_cpus();
        const auto policy = trainer_policy::all_epochs;

        for (auto optimizer : batch_optimizers)
        {
                const auto optname = "batch-" + optimizer;
                const auto params = to_params("opt", optimizer, "epochs", epochs, "policy", policy);
                evaluate_trainer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        const auto trainer = get_trainers().get("batch", params);
                        return trainer->train(task, fold, nthreads, loss, criterion, model);
                });
        }

        for (auto optimizer : stoch_optimizers)
        {
                const auto optname = "stoch-" + optimizer;
                const auto params = to_params("opt", optimizer, "epochs", epochs, "policy", policy);
                evaluate_trainer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        const auto trainer = get_trainers().get("stoch", params);
                        return trainer->train(task, fold, nthreads, loss, criterion, model);
                });
        }
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("benchmark trainers");
        cmdline.add("s", "samples",             "number of samples to use [100, 100000]", "10000");
        cmdline.add("", "mlps",                 "use MLPs with varying number of hidden layers");
        cmdline.add("", "mlp0",                 "use MLPs with 0 hidden layers");
        cmdline.add("", "mlp1",                 "use MLPs with 1 hidden layer");
        cmdline.add("", "mlp2",                 "use MLPs with 2 hidden layers");
        cmdline.add("", "mlp3",                 "use MLPs with 3 hidden layers");
        cmdline.add("", "convnets",             "use convolution networks with varying number of convolution layers");
        cmdline.add("", "convnet1",             "use convolution networks with 1 convolution layer");
        cmdline.add("", "convnet2",             "use convolution networks with 2 convolution layers");
        cmdline.add("", "convnet3",             "use convolution networks with 3 convolution layers");
        cmdline.add("", "batch",                "evaluate batch optimizers");
        cmdline.add("", "batch-gd",             "evaluate batch optimizer GD (gradient descent)");
        cmdline.add("", "batch-cgd",            "evaluate batch optimizer CGD (conjugate gradient descent)");
        cmdline.add("", "batch-lbfgs",          "evaluate batch optimizer LBFGS");
        cmdline.add("", "stoch",                "evaluate stoch optimizers");
        cmdline.add("", "stoch-sg",             "evaluate stoch optimizer SG (stoch gradient)");
        cmdline.add("", "stoch-ngd",            "evaluate stoch optimizer NGS (normalized gradient descent)");
        cmdline.add("", "stoch-sgm",            "evaluate stoch optimizer SGM (stoch gradient with momentum)");
        cmdline.add("", "stoch-ag",             "evaluate stoch optimizer AG (Nesterov's accelerated gradient)");
        cmdline.add("", "stoch-agfr",           "evaluate stoch optimizer AG (AG + function value restarts)");
        cmdline.add("", "stoch-aggr",           "evaluate stoch optimizer AG (AG + gradient restarts)");
        cmdline.add("", "stoch-adam",           "evaluate stoch optimizer ADAM");
        cmdline.add("", "stoch-adagrad",        "evaluate stoch optimizer ADAGRAD");
        cmdline.add("", "stoch-adadelta",       "evaluate stoch optimizer ADADELTA");
        cmdline.add("", "loss",                 "loss function (" + nano::concatenate(get_losses().ids()) + ")", "classnll");
        cmdline.add("", "criterion",            "training criterion (" + nano::concatenate(get_criteria().ids()) + ")", "avg");
        cmdline.add("", "activation",           "activation layer (act-unit, act-tanh, act-splus, act-snorm)", "act-snorm");
        cmdline.add("", "trials",               "number of models to train & evaluate", "10");
        cmdline.add("", "epochs",               "number of epochs", "100");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto count = nano::clamp(cmdline.get<size_t>("samples"), 100, 100 * 1000);
        const bool use_mlps = cmdline.has("mlps");
        const bool use_mlp0 = cmdline.has("mlp0");
        const bool use_mlp1 = cmdline.has("mlp1");
        const bool use_mlp2 = cmdline.has("mlp2");
        const bool use_mlp3 = cmdline.has("mlp3");
        const bool use_convnets = cmdline.has("convnets");
        const bool use_convnet1 = cmdline.has("convnet1");
        const bool use_convnet2 = cmdline.has("convnet2");
        const bool use_convnet3 = cmdline.has("convnet3");
        const auto cmd_loss = cmdline.get("loss");
        const auto cmd_criterion = cmdline.get("criterion");
        const auto activation = cmdline.get("activation");
        const auto trials = cmdline.get<size_t>("trials");
        const auto epochs = cmdline.get<size_t>("epochs");

        if (    !use_mlps &&
                !use_mlp0 &&
                !use_mlp1 &&
                !use_mlp2 &&
                !use_mlp3 &&
                !use_convnets &&
                !use_convnet1 &&
                !use_convnet2 &&
                !use_convnet3)
        {
                cmdline.usage();
        }

        strings_t batch_optimizers;
        if (cmdline.has("batch") || cmdline.has("batch-gd")) batch_optimizers.push_back("gd");
        if (cmdline.has("batch") || cmdline.has("batch-cgd")) batch_optimizers.push_back("cgd");
        if (cmdline.has("batch") || cmdline.has("batch-lbfgs")) batch_optimizers.push_back("lbfgs");

        strings_t stoch_optimizers;
        if (cmdline.has("stoch") || cmdline.has("stoch-sg")) stoch_optimizers.push_back("sg");
        if (cmdline.has("stoch") || cmdline.has("stoch-ngd")) stoch_optimizers.push_back("ngd");
        if (cmdline.has("stoch") || cmdline.has("stoch-sgm")) stoch_optimizers.push_back("sgm");
        if (cmdline.has("stoch") || cmdline.has("stoch-ag")) stoch_optimizers.push_back("ag");
        if (cmdline.has("stoch") || cmdline.has("stoch-agfr")) stoch_optimizers.push_back("agfr");
        if (cmdline.has("stoch") || cmdline.has("stoch-aggr")) stoch_optimizers.push_back("aggr");
        if (cmdline.has("stoch") || cmdline.has("stoch-adam")) stoch_optimizers.push_back("adam");
        if (cmdline.has("stoch") || cmdline.has("stoch-adagrad")) stoch_optimizers.push_back("adagrad");
        if (cmdline.has("stoch") || cmdline.has("stoch-adadelta")) stoch_optimizers.push_back("adadelta");

        if (    batch_optimizers.empty() &&
                stoch_optimizers.empty())
        {
                cmdline.usage();
        }

        // create task
        const size_t rows = 16;
        const size_t cols = 16;
        const color_mode color = color_mode::rgb;

        charset_task_t task(to_params(
                "type", charset_mode::digit, "color", color, "irows", rows, "icols", cols, "count", count));
        task.load();

        const size_t fold = 0;
        const auto outputs = task.osize();

        // construct models
        const auto mlp0 = string_t();
        const auto mlp1 = mlp0 + make_affine_layer(64, activation);
        const auto mlp2 = mlp1 + make_affine_layer(64, activation);
        const auto mlp3 = mlp2 + make_affine_layer(64, activation);

        const auto convnet0 = string_t();
        const auto convnet1 = convnet0 + make_conv_layer(32, 5, 5, 1, activation);
        const auto convnet2 = convnet1 + make_conv_layer(32, 5, 5, 4, activation);
        const auto convnet3 = convnet2 + make_conv_layer(32, 3, 3, 4, activation);

        const string_t outlayer = make_output_layer(outputs, activation);

        std::vector<std::pair<string_t, string_t>> networks;
        if (use_mlps || use_mlp0) networks.emplace_back(mlp0 + outlayer, "mlp0");
        if (use_mlps || use_mlp1) networks.emplace_back(mlp1 + outlayer, "mlp1");
        if (use_mlps || use_mlp2) networks.emplace_back(mlp2 + outlayer, "mlp2");
        if (use_mlps || use_mlp3) networks.emplace_back(mlp3 + outlayer, "mlp3");
        if (use_convnets || use_convnet1) networks.emplace_back(convnet1 + outlayer, "convnet1");
        if (use_convnets || use_convnet2) networks.emplace_back(convnet2 + outlayer, "convnet2");
        if (use_convnets || use_convnet3) networks.emplace_back(convnet3 + outlayer, "convnet3");

        // vary the model
        for (const auto& net : networks)
        {
                const auto& network = net.first;
                const auto& netname = net.second;

                log_info() << "<<< running network [" << network << "] ...";

                const auto model = get_models().get("forward-network", network);
                model->resize(task, true);

                // generate fixed random starting points
                vectors_t x0s(trials);
                for (vector_t& x0 : x0s)
                {
                        model->random_params();
                        model->save_params(x0);
                }

                log_info() << "<<< running loss [" << cmd_loss << "] ...";

                const auto loss = get_losses().get(cmd_loss);

                table_t table(netname + "-" + cmd_loss);
                table.header()
                        << "test criteria"
                        << "test error"
                        << "convergence speed"
                        << "time [sec]";

                // vary the criteria
                const auto criterion = get_criteria().get(cmd_criterion);
                const auto basename = "[" + cmd_criterion + "] ";
                const auto basepath = netname + "-" + cmd_loss + "-" + cmd_criterion + "-";

                evaluate(*model, task, fold, *loss, *criterion, x0s, epochs,
                         batch_optimizers, stoch_optimizers,
                         basename, basepath, table);

                // show results
                std::cout << table;
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
