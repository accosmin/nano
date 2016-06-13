#include "text/table.h"
#include "text/cmdline.h"
#include "cortex/cortex.h"
#include "cortex/logger.h"
#include "thread/thread.h"
#include "cortex/measure.hpp"
#include "text/to_params.hpp"
#include "optim/batch/types.h"
#include "optim/stoch/types.h"
#include "text/concatenate.hpp"
#include "cortex/tasks/task_charset.h"
#include "cortex/layers/make_layers.h"

using namespace nano;

template
<
        typename tvalue
>
static string_t stats_to_string(const stats_t<tvalue>& stats)
{
        return  to_string(static_cast<tvalue>(stats.avg()))
                + "+/-" + to_string(static_cast<tvalue>(stats.stdev()))
                + " [" + to_string(stats.min())
                + ", " + to_string(stats.max())
                + "]";
}

template
<
        typename ttrainer
>
static void test_optimizer(model_t& model, const string_t& name, const string_t& basepath,
        table_t& table, const vectors_t& x0s, const ttrainer& trainer)
{
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

                errors(opt_state.m_test.m_error_avg);
                speeds(opt_speed);
                timings(static_cast<scalar_t>(timer.seconds().count()));

                const auto path = basepath + "-trial" + to_string(i) + ".state";

                const auto opt_states = result.optimum_states();
                save(path, opt_states);
        }

        table.append(name)
                << stats_to_string(errors)
                << stats_to_string(speeds)
                << stats_to_string(timings);
}

static void evaluate(model_t& model,
        const task_t& task, const size_t fold,
        const loss_t& loss, const criterion_t& criterion, const vectors_t& x0s, const size_t iterations,
        const std::vector<batch_optimizer>& batch_optimizers,
        const std::vector<batch_optimizer>& minibatch_optimizers,
        const std::vector<stoch_optimizer>& stochastic_optimizers,
        const string_t& basename, const string_t& basepath, table_t& table)
{
        const auto epsilon = scalar_t(1e-6);
        const auto nthreads = thread::concurrency();
        const auto policy = trainer_policy::all_epochs;

        for (auto optimizer : batch_optimizers)
        {
                if (optimizer == batch_optimizer::LBFGS)
                {
                        for (const auto history : {4, 6, 8, 10})
                        {
                                const auto optname = "batch-" + to_string(optimizer) + "[" + to_string(history) + "]";
                                const auto params = to_params("opt", optimizer, "iters", iterations, "eps", epsilon, "history", history, "policy", policy);
                                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                                {
                                        const auto trainer = get_trainers().get("batch", params);
                                        return trainer->train(task, fold, nthreads, loss, criterion, model);
                                });
                        }
                }
                else
                {
                        const auto optname = "batch-" + to_string(optimizer);
                        const auto params = to_params("opt", optimizer, "iters", iterations, "eps", epsilon, "policy", policy);
                        test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                        {
                                const auto trainer = get_trainers().get("batch", params);
                                return trainer->train(task, fold, nthreads, loss, criterion, model);
                        });
                }
        }

        for (auto optimizer : minibatch_optimizers)
        {
                const auto optname = "minibatch-" + to_string(optimizer);
                const auto params = to_params("opt", optimizer, "epochs", iterations, "eps", epsilon, "policy", policy);
                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        const auto trainer = get_trainers().get("minibatch", params);
                        return trainer->train(task, fold, nthreads, loss, criterion, model);
                });
        }

        for (auto optimizer : stochastic_optimizers)
        {
                const auto optname = "stochastic-" + to_string(optimizer);
                const auto params = to_params("opt", optimizer, "epochs", iterations, "policy", policy);
                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        const auto trainer = get_trainers().get("stochastic", params);
                        return trainer->train(task, fold, nthreads, loss, criterion, model);
                });
        }
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("benchmark trainers");
        cmdline.add("", "mlps",                 "use MLPs with varying number of hidden layers");
        cmdline.add("", "convnets",             "use convolution networks with varying number of convolution layers");
        cmdline.add("", "batch",                "evaluate batch optimizers");
        cmdline.add("", "batch-gd",             "evaluate batch optimizer GD (gradient descent)");
        cmdline.add("", "batch-cgd",            "evaluate batch optimizer CGD (conjugate gradient descent)");
        cmdline.add("", "batch-lbfgs",          "evaluate batch optimizer LBFGS");
        cmdline.add("", "minibatch",            "evaluate mini-batch optimizers");
        cmdline.add("", "minibatch-gd",         "evaluate mini-batch optimizer GD (gradient descent)");
        cmdline.add("", "minibatch-cgd",        "evaluate mini-batch optimizer CGD (conjugate gradient descent)");
        cmdline.add("", "minibatch-lbfgs",      "evaluate mini-batch optimizer LBFGS");
        cmdline.add("", "stochastic",           "evaluate stochastic optimizers");
        cmdline.add("", "stochastic-sg",        "evaluate stochastic optimizer SG (stochastic gradient)");
        cmdline.add("", "stochastic-ngd",       "evaluate stochastic optimizer NGS (normalized gradient descent)");
        cmdline.add("", "stochastic-sgm",       "evaluate stochastic optimizer SGM (stochastic gradient with momentum)");
        cmdline.add("", "stochastic-ag",        "evaluate stochastic optimizer AG (Nesterov's accelerated gradient)");
        cmdline.add("", "stochastic-agfr",      "evaluate stochastic optimizer AG (AG + function value restarts)");
        cmdline.add("", "stochastic-aggr",      "evaluate stochastic optimizer AG (AG + gradient restarts)");
        cmdline.add("", "stochastic-adam",      "evaluate stochastic optimizer ADAM");
        cmdline.add("", "stochastic-adagrad",   "evaluate stochastic optimizer ADAGRAD");
        cmdline.add("", "stochastic-adadelta",  "evaluate stochastic optimizer ADADELTA");
        cmdline.add("", "l2n-reg",              "also evaluate the l2-norm-based regularizer");
        cmdline.add("", "var-reg",              "also evaluate the variance-based regularizer");
        cmdline.add("", "trials",               "number of models to train & evaluate", "10");
        cmdline.add("", "iterations",           "number of iterations/epochs", "64");

        cmdline.process(argc, argv);

        // check arguments and options
        const bool use_mlps = cmdline.has("mlps");
        const bool use_convnets = cmdline.has("convnets");
        const bool use_reg_l2n = cmdline.has("l2n-reg");
        const bool use_reg_var = cmdline.has("var-reg");
        const auto trials = cmdline.get<size_t>("trials");
        const auto iterations = cmdline.get<size_t>("iterations");

        if (    !use_mlps &&
                !use_convnets)
        {
                cmdline.usage();
        }

        std::vector<batch_optimizer> batch_optimizers;
        if (cmdline.has("batch") || cmdline.has("batch-gd")) batch_optimizers.push_back(batch_optimizer::GD);
        if (cmdline.has("batch") || cmdline.has("batch-cgd")) batch_optimizers.push_back(batch_optimizer::CGD);
        if (cmdline.has("batch") || cmdline.has("batch-lbfgs")) batch_optimizers.push_back(batch_optimizer::LBFGS);

        std::vector<batch_optimizer> minibatch_optimizers;
        if (cmdline.has("minibatch") || cmdline.has("minibatch-gd")) minibatch_optimizers.push_back(batch_optimizer::GD);
        if (cmdline.has("minibatch") || cmdline.has("minibatch-cgd")) minibatch_optimizers.push_back(batch_optimizer::CGD);
        if (cmdline.has("minibatch") || cmdline.has("minibatch-lbfgs")) minibatch_optimizers.push_back(batch_optimizer::LBFGS);

        std::vector<stoch_optimizer> stochastic_optimizers;
        if (cmdline.has("stochastic") || cmdline.has("stochastic-sg")) stochastic_optimizers.push_back(stoch_optimizer::SG);
        if (cmdline.has("stochastic") || cmdline.has("stochastic-ngd")) stochastic_optimizers.push_back(stoch_optimizer::NGD);
        if (cmdline.has("stochastic") || cmdline.has("stochastic-sgm")) stochastic_optimizers.push_back(stoch_optimizer::SGM);
        if (cmdline.has("stochastic") || cmdline.has("stochastic-ag")) stochastic_optimizers.push_back(stoch_optimizer::AG);
        if (cmdline.has("stochastic") || cmdline.has("stochastic-agfr")) stochastic_optimizers.push_back(stoch_optimizer::AGFR);
        if (cmdline.has("stochastic") || cmdline.has("stochastic-aggr")) stochastic_optimizers.push_back(stoch_optimizer::AGGR);
        if (cmdline.has("stochastic") || cmdline.has("stochastic-adam")) stochastic_optimizers.push_back(stoch_optimizer::ADAM);
        if (cmdline.has("stochastic") || cmdline.has("stochastic-adagrad")) stochastic_optimizers.push_back(stoch_optimizer::ADAGRAD);
        if (cmdline.has("stochastic") || cmdline.has("stochastic-adadelta")) stochastic_optimizers.push_back(stoch_optimizer::ADADELTA);

        if (    batch_optimizers.empty() &&
                minibatch_optimizers.empty() &&
                stochastic_optimizers.empty())
        {
                cmdline.usage();
        }

        // create task
        const size_t rows = 16;
        const size_t cols = 16;
        const size_t count = thread::concurrency() * 32 * 20;
        const color_mode color = color_mode::rgb;

        charset_task_t task(charset::digit, color, rows, cols, count);
        task.load();

        const size_t fold = 0;
        const auto outputs = task.osize();

        // construct models
        const auto activation = "act-snorm";
        const auto pooling = "pool-full";

        const auto mlp0 = string_t();
        const auto mlp1 = mlp0 + make_affine_layer(64, activation);
        const auto mlp2 = mlp1 + make_affine_layer(64, activation);
        const auto mlp3 = mlp2 + make_affine_layer(64, activation);

        const auto convnet1 =
                make_conv_pool_layer(32, 7, 7, 1, activation, pooling);

        const auto convnet2 =
                make_conv_pool_layer(32, 7, 7, 1, activation, pooling) +
                make_conv_layer(64, 5, 5, 4);

        const auto convnet3 =
                make_conv_layer(32, 7, 7, 1, activation) +
                make_conv_layer(64, 5, 5, 4, activation) +
                make_conv_layer(128, 3, 3, 8, activation);

        const string_t outlayer = make_output_layer(outputs, activation);

        std::vector<std::pair<string_t, string_t>> networks;
        if (use_mlps)
        {
                networks.emplace_back(mlp0 + outlayer, "mlp0");
                networks.emplace_back(mlp1 + outlayer, "mlp1");
                networks.emplace_back(mlp2 + outlayer, "mlp2");
                networks.emplace_back(mlp3 + outlayer, "mlp3");
        }
        if (use_convnets)
        {
                networks.emplace_back(convnet1 + outlayer, "convnet1");
                networks.emplace_back(convnet2 + outlayer, "convnet2");
                networks.emplace_back(convnet3 + outlayer, "convnet3");
        }

        const strings_t losses = { "classnll" }; //get_losses().ids();

        strings_t criteria;
        criteria.push_back("avg"); //get_criteria().ids();
        if (use_reg_l2n) { criteria.push_back("l2n-reg"); }
        if (use_reg_var) { criteria.push_back("var-reg"); }

        // vary the model
        for (const auto& net : networks)
        {
                const auto& network = net.first;
                const auto& netname = net.second;

                log_info() << "<<< running network [" << network << "] ...";

                const auto model = get_models().get("forward-network", network);
                model->resize(task, false);

                // generate fixed random starting points
                vectors_t x0s(trials);
                for (vector_t& x0 : x0s)
                {
                        model->random_params();
                        model->save_params(x0);
                }

                // vary the loss
                for (const string_t& iloss : losses)
                {
                        log_info() << "<<< running loss [" << iloss << "] ...";

                        const auto loss = get_losses().get(iloss);

                        table_t table(netname + "-" + iloss);
                        table.header() << "test error"
                                       << "convergence speed"
                                       << "time [sec]";

                        // vary the criteria
                        for (const string_t& icriterion : criteria)
                        {
                                const auto criterion = get_criteria().get(icriterion);
                                const auto basename = "[" + icriterion + "] ";
                                const auto basepath = netname + "-" + iloss + "-" + icriterion + "-";

                                evaluate(*model, task, fold, *loss, *criterion, x0s, iterations,
                                         batch_optimizers, minibatch_optimizers, stochastic_optimizers,
                                         basename, basepath, table);
                        }

                        // show results
                        table.print(std::cout);
                }

                log_info();
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
