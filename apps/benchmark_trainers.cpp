#include "text/table.h"
#include "text/cmdline.h"
#include "cortex/batch.h"
#include "cortex/cortex.h"
#include "thread/thread.h"
#include "cortex/minibatch.h"
#include "cortex/stochastic.h"
#include "cortex/util/logger.h"
#include "cortex/accumulator.h"
#include "text/concatenate.hpp"
#include "cortex/util/measure.hpp"
#include "cortex/tasks/task_charset.h"
#include "cortex/layers/make_layers.h"

using namespace zob;

template
<
        typename tvalue
>
static string_t stats_to_string(const zob::stats_t<tvalue>& stats)
{
        return  zob::to_string(static_cast<tvalue>(stats.avg()))
                + "+/-" + zob::to_string(static_cast<tvalue>(stats.stdev()))
                + " [" + zob::to_string(stats.min())
                + ", " + zob::to_string(stats.max())
                + "]";
}

template
<
        typename ttrainer
>
static void test_optimizer(model_t& model, const string_t& name, const string_t& basepath,
        zob::table_t& table, const vectors_t& x0s, const ttrainer& trainer)
{
        zob::stats_t<scalar_t> terrors;
        zob::stats_t<scalar_t> verrors;
        zob::stats_t<scalar_t> speeds;
        zob::stats_t<scalar_t> timings;

        log_info() << "<<< running " << name << " ...";

        for (size_t i = 0; i < x0s.size(); ++ i)
        {
                const zob::timer_t timer;

                model.load_params(x0s[i]);

                const auto result = trainer();
                const auto opt_state = result.optimum_state();
                const auto opt_speed = zob::convergence_speed(result.optimum_states());

                terrors(opt_state.m_terror_avg);
                verrors(opt_state.m_verror_avg);
                speeds(opt_speed);
                timings(static_cast<scalar_t>(timer.seconds().count()));

                log_info() << "<<< " << name
                           << ", optimum = " << result.optimum_config()
                           << "epoch=" << result.optimum_epoch()
                           << ", train = " << opt_state.m_terror_avg
                           << ", valid = " << opt_state.m_verror_avg
                           << ", speed = " << opt_speed << "/s"
                           << " done in " << timer.elapsed() << ".";

                const auto path = basepath + "-trial" + zob::to_string(i) + ".state";

                const auto opt_states = result.optimum_states();
                zob::save(path, opt_states);
        }

        table.append(name)
                << stats_to_string(terrors)
                << stats_to_string(verrors)
                << stats_to_string(speeds)
                << stats_to_string(timings);
}

static void evaluate(
        model_t& model, const task_t& task, const fold_t& fold, const loss_t& loss, const criterion_t& criterion,
        const size_t trials, const size_t iterations, const string_t& basepath, zob::table_t& table)
{
        const scalar_t epsilon = 1e-4;
        const size_t n_threads = zob::n_threads();
        const bool verbose = true;

        // generate fixed random starting points
        vectors_t x0s(trials);
        for (vector_t& x0 : x0s)
        {
                model.random_params();
                model.save_params(x0);
        }

        // batch optimizers
        const auto batch_optimizers =
        {
                zob::batch_optimizer::GD,
                zob::batch_optimizer::CGD,
                zob::batch_optimizer::LBFGS
        };

        // minibatch optimizers
        const auto minibatch_optimizers =
        {
                zob::batch_optimizer::GD,
                zob::batch_optimizer::CGD,
                zob::batch_optimizer::LBFGS
        };

        // stochastic optimizers
        const auto stoch_optimizers =
        {
                zob::stoch_optimizer::SG,
                zob::stoch_optimizer::SGM,
                zob::stoch_optimizer::AG,
                zob::stoch_optimizer::AGFR,
                zob::stoch_optimizer::AGGR,
                zob::stoch_optimizer::ADAGRAD,
                zob::stoch_optimizer::ADADELTA,
                zob::stoch_optimizer::ADAM
        };

        const string_t basename = "[" + criterion.description() + "] ";

        // run optimizers and collect results
        for (zob::batch_optimizer optimizer : batch_optimizers)
        {
                const auto optname = "batch-" + zob::to_string(optimizer);
                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        return  zob::batch_train(
                                model, task, fold, n_threads, loss, criterion, optimizer, iterations, epsilon, verbose);
                });
        }

        for (zob::batch_optimizer optimizer : minibatch_optimizers)
        {
                const auto optname = "minibatch-" + zob::to_string(optimizer);
                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        return  zob::minibatch_train(
                                model, task, fold, n_threads, loss, criterion, optimizer, iterations, epsilon, verbose);
                });
        }

        for (zob::stoch_optimizer optimizer : stoch_optimizers)
        {
                const auto optname = "stochastic-" + zob::to_string(optimizer);
                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        return  zob::stochastic_train(
                                model, task, fold, n_threads, loss, criterion, optimizer, iterations, verbose);
                });
        }
}

int main(int argc, char* argv[])
{
        zob::init();

        using namespace zob;

        // parse the command line
        zob::cmdline_t cmdline("benchmark trainers");
        cmdline.add("", "l2n-reg",      "also evaluate the l2-norm-based regularizer");
        cmdline.add("", "var-reg",      "also evaluate the variance-based regularizer");
        cmdline.add("", "mlp0",         "MLP with 0 hidden layers");
        cmdline.add("", "mlp1",         "MLP with 1 hidden layers");
        cmdline.add("", "mlp2",         "MLP with 2 hidden layers");
        cmdline.add("", "mlp3",         "MLP with 3 hidden layers");
        cmdline.add("", "convnet",      "fully-connected convolution network");
        cmdline.add("", "trials",       "number of models to train & evaluate", "10");
        cmdline.add("", "iterations",   "number of iterations/epochs", "64");

        cmdline.process(argc, argv);

        // check arguments and options
        const bool use_reg_l2n = cmdline.has("l2n-reg");
        const bool use_reg_var = cmdline.has("var-reg");
        const bool use_mlp0 = cmdline.has("mlp0");
        const bool use_mlp1 = cmdline.has("mlp1");
        const bool use_mlp2 = cmdline.has("mlp2");
        const bool use_mlp3 = cmdline.has("mlp3");
        const bool use_convnet = cmdline.has("convnet");
        const auto trials = cmdline.get<size_t>("trials");
        const auto iterations = cmdline.get<size_t>("iterations");

        if (    !use_mlp0 &&
                !use_mlp1 &&
                !use_mlp2 &&
                !use_mlp3 &&
                !use_convnet)
        {
                cmdline.usage();
        }

        // create task
        const size_t rows = 16;
        const size_t cols = 16;
        const size_t samples = zob::n_threads() * 256 * 10;
        const color_mode color = color_mode::rgba;

        charset_task_t task(charset::numeric, rows, cols, color, samples);
        task.load("");
        task.describe();

        const auto fold = std::make_pair(0, protocol::train);

        const auto outputs = task.osize();

        // construct models
        const string_t mlp0;
        const string_t mlp1 = mlp0 + make_affine_layer(16);
        const string_t mlp2 = mlp1 + make_affine_layer(16);
        const string_t mlp3 = mlp2 + make_affine_layer(16);

        const string_t convnet =
                make_conv_pool_layer(16, 7, 7) +
                make_conv_layer(16, 5, 5);

        const string_t outlayer = make_output_layer(outputs);

        std::vector<std::pair<string_t, string_t>> networks;
        if (use_mlp0) { networks.emplace_back(mlp0 + outlayer, "mlp0"); }
        if (use_mlp1) { networks.emplace_back(mlp1 + outlayer, "mlp1"); }
        if (use_mlp2) { networks.emplace_back(mlp2 + outlayer, "mlp2"); }
        if (use_mlp3) { networks.emplace_back(mlp3 + outlayer, "mlp3"); }
        if (use_convnet) { networks.emplace_back(convnet + outlayer, "convnet"); }

        const strings_t losses = { "classnll" }; //zob::get_losses().ids();

        strings_t criteria;
        criteria.push_back("avg"); //zob::get_criteria().ids();
        if (use_reg_l2n) { criteria.push_back("l2n-reg"); }
        if (use_reg_var) { criteria.push_back("var-reg"); }

        // vary the model
        for (const auto& net : networks)
        {
                const auto& network = net.first;
                const auto& netname = net.second;

                log_info() << "<<< running network [" << network << "] ...";

                const auto model = zob::get_models().get("forward-network", network);
                model->resize(task, true);

                // vary the loss
                for (const string_t& iloss : losses)
                {
                        log_info() << "<<< running loss [" << iloss << "] ...";

                        const auto loss = zob::get_losses().get(iloss);

                        zob::table_t table("optimizer");
                        table.header() << "train error"
                                       << "valid error"
                                       << "convergence speed"
                                       << "time [sec]";

                        // vary the criteria
                        for (const string_t& icriterion : criteria)
                        {
                                const auto criterion = zob::get_criteria().get(icriterion);

                                const auto basepath = netname + "-" + iloss + "-" + icriterion + "-";

                                evaluate(*model, task, fold, *loss, *criterion, trials, iterations, basepath, table);
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
