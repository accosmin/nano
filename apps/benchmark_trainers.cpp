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

using namespace cortex;

template
<
        typename tvalue
>
static string_t stats_to_string(const math::stats_t<tvalue>& stats)
{
        return  text::to_string(static_cast<tvalue>(stats.avg()))
                + "+/-" + text::to_string(static_cast<tvalue>(stats.stdev()))
                + " [" + text::to_string(stats.min())
                + ", " + text::to_string(stats.max())
                + "]";
}

template
<
        typename ttrainer
>
static void test_optimizer(model_t& model, const string_t& name, const string_t& basepath,
        text::table_t& table, const vectors_t& x0s, const ttrainer& trainer)
{
        math::stats_t<scalar_t> terrors;
        math::stats_t<scalar_t> verrors;
        math::stats_t<scalar_t> speeds;
        math::stats_t<scalar_t> timings;

        log_info() << "<<< running " << name << " ...";

        for (size_t i = 0; i < x0s.size(); ++ i)
        {
                const cortex::timer_t timer;

                model.load_params(x0s[i]);

                const auto result = trainer();
                const auto opt_state = result.optimum_state();
                const auto opt_speed = cortex::convergence_speed(result.optimum_states());

                terrors(opt_state.m_terror_avg);
                verrors(opt_state.m_verror_avg);
                speeds(opt_speed);
                timings(static_cast<scalar_t>(timer.seconds().count()));

                log_info() << "<<< " << name
                           << ", optimum = {" << text::concatenate(result.optimum_config())
                           << "}/" << result.optimum_epoch()
                           << ", train = " << opt_state.m_terror_avg
                           << ", valid = " << opt_state.m_verror_avg
                           << ", speed = " << opt_speed << "/s"
                           << " done in " << timer.elapsed() << ".";

                const auto path = basepath + "-trial" + text::to_string(i) + ".state";

                const auto opt_states = result.optimum_states();
                cortex::save(path, opt_states);
        }

        table.append(name)
                << stats_to_string(terrors)
                << stats_to_string(verrors)
                << stats_to_string(speeds)
                << stats_to_string(timings);
}

static void test_optimizers(
        const task_t& task, model_t& model, const sampler_t& tsampler, const sampler_t& vsampler,
        const loss_t& loss, const criterion_t& criterion,
        const size_t trials, const size_t iterations, const string_t& basepath, text::table_t& table)
{
        const size_t batch_iterations = iterations;
        const size_t minibatch_epochs = iterations;
        const size_t stochastic_epochs = iterations;
        const scalar_t epsilon = 1e-4;

        const size_t n_threads = thread::n_threads();
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
                math::batch_optimizer::GD,
                math::batch_optimizer::CGD,
                math::batch_optimizer::LBFGS
        };

        // minibatch optimizers
        const auto minibatch_optimizers =
        {
                math::batch_optimizer::GD,
                math::batch_optimizer::CGD,
                math::batch_optimizer::LBFGS
        };

        // stochastic optimizers
        const auto stoch_optimizers =
        {
                math::stoch_optimizer::SG,
                math::stoch_optimizer::SGM,
                math::stoch_optimizer::AG,
                math::stoch_optimizer::AGFR,
                math::stoch_optimizer::AGGR,
                math::stoch_optimizer::ADAGRAD,
                math::stoch_optimizer::ADADELTA,
                math::stoch_optimizer::ADAM
        };

        const string_t basename = "[" + criterion.description() + "] ";

        // run optimizers and collect results
        for (math::batch_optimizer optimizer : batch_optimizers)
        {
                const auto optname = "batch-" + text::to_string(optimizer);
                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        return cortex::batch_train(
                                model, task, tsampler, vsampler, n_threads,
                                loss, criterion, optimizer, batch_iterations, epsilon, verbose);
                });
        }

        for (math::batch_optimizer optimizer : minibatch_optimizers)
        {
                const auto optname = "minibatch-" + text::to_string(optimizer);
                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        return cortex::minibatch_train(
                                model, task, tsampler, vsampler, n_threads,
                                loss, criterion, optimizer, minibatch_epochs, epsilon, verbose);
                });
        }

        for (math::stoch_optimizer optimizer : stoch_optimizers)
        {
                const auto optname = "stochastic-" + text::to_string(optimizer);
                test_optimizer(model, basename + optname, basepath + optname, table, x0s, [&] ()
                {
                        return cortex::stochastic_train(
                                model, task, tsampler, vsampler, n_threads,
                                loss, criterion, optimizer, stochastic_epochs, verbose);
                });
        }
}

int main(int argc, char* argv[])
{
        cortex::init();

        using namespace cortex;

        // parse the command line
        text::cmdline_t cmdline("benchmark trainers");
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
        const size_t samples = thread::n_threads() * 256 * 10;
        const color_mode color = color_mode::rgba;

        charset_task_t task(charset::numeric, rows, cols, color, samples);
        task.load("");
        task.describe();

        const auto outputs = task.osize();

        // create training & validation samples
        sampler_t tsampler(task.samples());
        tsampler.push(cortex::annotation::annotated);

        sampler_t vsampler(task.samples());
        tsampler.split(80, vsampler);

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

        const strings_t losses = { "classnll" }; //cortex::get_losses().ids();

        strings_t criteria;
        criteria.push_back("avg"); //cortex::get_criteria().ids();
        if (use_reg_l2n) { criteria.push_back("l2n-reg"); }
        if (use_reg_var) { criteria.push_back("var-reg"); }

        // vary the model
        for (const auto& net : networks)
        {
                const auto& network = net.first;
                const auto& netname = net.second;

                log_info() << "<<< running network [" << network << "] ...";

                const auto model = cortex::get_models().get("forward-network", network);
                model->resize(task, true);

                // vary the loss
                for (const string_t& iloss : losses)
                {
                        log_info() << "<<< running loss [" << iloss << "] ...";

                        const auto loss = cortex::get_losses().get(iloss);

                        text::table_t table("optimizer");
                        table.header() << "train error"
                                       << "valid error"
                                       << "convergence speed"
                                       << "time [sec]";

                        // vary the criteria
                        for (const string_t& icriterion : criteria)
                        {
                                const auto criterion = cortex::get_criteria().get(icriterion);

                                const auto basepath = netname + "-" + iloss + "-" + icriterion + "-";

                                test_optimizers(task, *model, tsampler, vsampler, *loss, *criterion,
                                                trials, iterations, basepath, table);
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
