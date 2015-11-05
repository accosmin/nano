#include "cortex/batch.h"
#include "cortex/cortex.h"
#include "thread/thread.h"
#include "cortex/minibatch.h"
#include "cortex/util/table.h"
#include "cortex/stochastic.h"
#include "cortex/util/logger.h"
#include "cortex/accumulator.h"
#include "text/concatenate.hpp"
#include "cortex/util/measure.hpp"
#include "cortex/tasks/task_charset.h"
#include <boost/program_options.hpp>

namespace
{
        using namespace cortex;

        string_t stats_to_string(const math::stats_t<scalar_t>& stats)
        {
                return  text::to_string(stats.avg())
                        + "+/-" + text::to_string(stats.stdev())
                        + " [" + text::to_string(stats.min())
                        + ", " + text::to_string(stats.max())
                        + "]";
        }

        template
        <
                typename ttrainer
        >
        void test_optimizer(model_t& model, const string_t& name, table_t& table, const vectors_t& x0s,
                const ttrainer& trainer)
        {
                math::stats_t<scalar_t> terrors;
                math::stats_t<scalar_t> verrors;
                math::stats_t<size_t> timings;

                log_info() << "<<< running " << name << " ...";

                for (const vector_t& x0 : x0s)
                {
                        const cortex::timer_t timer;

                        model.load_params(x0);

                        const trainer_result_t result = trainer();
                        const trainer_state_t state = result.optimum_state();

                        terrors(state.m_terror_avg);
                        verrors(state.m_verror_avg);
                        timings(timer.seconds());

                        log_info() << "<<< " << name
                                   << ", optimum = {" << text::concatenate(result.optimum_config())
                                   << "}/" << result.optimum_epoch()
                                   << ", train = " << state.m_terror_avg
                                   << ", valid = " << state.m_verror_avg
                                   << " done in " << timer.elapsed() << ".";
                }

                table.append(name)
                        << stats_to_string(terrors)
                        << stats_to_string(verrors)
                        << static_cast<size_t>(timings.avg());
        }

        void test_optimizers(
                const task_t& task, model_t& model, const sampler_t& tsampler, const sampler_t& vsampler,
                const loss_t& loss, const string_t& criterion, table_t& table)
        {
                const size_t cmd_trials = 10;

                const size_t cmd_iterations = 64;
                const size_t cmd_minibatch_epochs = cmd_iterations;
                const size_t cmd_stochastic_epochs = cmd_iterations;
                const scalar_t cmd_epsilon = 1e-4;

                const size_t n_threads = thread::n_threads();
                const bool verbose = true;

                // generate fixed random starting points
                vectors_t x0s(cmd_trials);
                for (vector_t& x0 : x0s)
                {
                        model.random_params();
                        model.save_params(x0);
                }

                // batch optimizers
                const auto batch_optimizers =
                {
                        min::batch_optimizer::GD,
                        min::batch_optimizer::CGD,
                        min::batch_optimizer::LBFGS
                };

                // minibatch optimizers
                const auto minibatch_optimizers =
                {
                        min::batch_optimizer::GD,
                        min::batch_optimizer::CGD,
                        min::batch_optimizer::LBFGS
                };

                // stochastic optimizers
                const auto stoch_optimizers =
                {
                        min::stoch_optimizer::SG,
                        min::stoch_optimizer::SGA,
                        min::stoch_optimizer::SIA,
                        min::stoch_optimizer::AG,
                        min::stoch_optimizer::AGGR,
                        min::stoch_optimizer::ADAGRAD,
                        min::stoch_optimizer::ADADELTA
                };

                const string_t basename = "[" + text::to_string(criterion) + "] ";

                // run optimizers and collect results
                for (min::batch_optimizer optimizer : batch_optimizers)
                {
                        test_optimizer(model, basename + "batch-" + text::to_string(optimizer), table, x0s, [&] ()
                        {
                                return cortex::batch_train(
                                        model, task, tsampler, vsampler, n_threads,
                                        loss, criterion, optimizer, cmd_iterations, cmd_epsilon, verbose);
                        });
                }

                for (min::batch_optimizer optimizer : minibatch_optimizers)
                {
                        test_optimizer(model, basename + "minibatch-" + text::to_string(optimizer), table, x0s, [&] ()
                        {
                                return cortex::minibatch_train(
                                        model, task, tsampler, vsampler, n_threads,
                                        loss, criterion, optimizer, cmd_minibatch_epochs, cmd_epsilon, verbose);
                        });
                }

                for (min::stoch_optimizer optimizer : stoch_optimizers)
                {
                        test_optimizer(model, basename + "stochastic-" + text::to_string(optimizer), table, x0s, [&] ()
                        {
                                return cortex::stochastic_train(
                                        model, task, tsampler, vsampler, n_threads,
                                        loss, criterion, optimizer, cmd_stochastic_epochs, verbose);
                        });
                }
        }
}

int main(int argc, char* argv[])
{
        cortex::init();

        using namespace cortex;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "benchmark trainers");
        po_desc.add_options()("mlp0", "MLP with 0 hidden layers");
        po_desc.add_options()("mlp1", "MLP with 1 hidden layers");
        po_desc.add_options()("mlp2", "MLP with 2 hidden layers");
        po_desc.add_options()("mlp3", "MLP with 3 hidden layers");
        po_desc.add_options()("conv1", "convolution network with 1 convolution layers");
        po_desc.add_options()("conv2", "convolution network with 2 convolution layers");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);

        // check arguments and options
        if (	po_vm.empty() ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const bool use_mlp0 = po_vm.count("mlp0");
        const bool use_mlp1 = po_vm.count("mlp1");
        const bool use_mlp2 = po_vm.count("mlp2");
        const bool use_mlp3 = po_vm.count("mlp3");
        const bool use_conv1 = po_vm.count("conv1");
        const bool use_conv2 = po_vm.count("conv2");

        if (    !use_mlp0 &&
                !use_mlp1 &&
                !use_mlp2 &&
                !use_mlp3 &&
                !use_conv1 &&
                !use_conv2)
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        // create task
        const size_t cmd_rows = 16;
        const size_t cmd_cols = 16;
        const size_t cmd_samples = thread::n_threads() * 256 * 10;
        const color_mode cmd_color = color_mode::rgba;

        charset_task_t task(charset::numeric, cmd_rows, cmd_cols, cmd_color, cmd_samples);
        task.load("");
	task.describe();

        const auto cmd_outputs = task.osize();

        // create training & validation samples
        sampler_t tsampler(task.samples());
        tsampler.push(cortex::annotation::annotated);

        sampler_t vsampler(task.samples());
        tsampler.split(80, vsampler);

        // construct models
        const string_t lmodel0;
        const string_t lmodel1 = lmodel0 + "linear:dims=16;act-snorm;";
        const string_t lmodel2 = lmodel1 + "linear:dims=16;act-snorm;";
        const string_t lmodel3 = lmodel2 + "linear:dims=16;act-snorm;";

        const string_t cmodel0;
        const string_t cmodel1 = cmodel0 + "conv:dims=16,rows=7,cols=7;pool-max;act-snorm;";
        const string_t cmodel2 = cmodel1 + "conv:dims=16,rows=5,cols=5;act-snorm;";

        const string_t outlayer = "linear:dims=" + text::to_string(cmd_outputs) + ";";

        strings_t cmd_networks;
        if (use_mlp0) { cmd_networks.push_back(lmodel0 + outlayer); }
        if (use_mlp1) { cmd_networks.push_back(lmodel1 + outlayer); }
        if (use_mlp2) { cmd_networks.push_back(lmodel2 + outlayer); }
        if (use_mlp3) { cmd_networks.push_back(lmodel3 + outlayer); }
        if (use_conv1) { cmd_networks.push_back(cmodel1 + outlayer); }
        if (use_conv2) { cmd_networks.push_back(cmodel2 + outlayer); }

        const strings_t cmd_losses = { "classnll" }; //cortex::get_losses().ids();
        const strings_t cmd_criteria = cortex::get_criteria().ids();

        // vary the model
        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                const rmodel_t model = cortex::get_models().get("forward-network", cmd_network);
                assert(model);
                model->resize(task, true);

                // vary the loss
                for (const string_t& cmd_loss : cmd_losses)
                {
                        log_info() << "<<< running loss [" << cmd_loss << "] ...";

                        const rloss_t loss = cortex::get_losses().get(cmd_loss);
                        assert(loss);

                        table_t table("optimizer");
                        table.header() << "train error"
                                       << "valid error"
                                       << "time [sec]";

                        // vary the criteria
                        for (const string_t& cmd_criterion : cmd_criteria)
                        {
                                test_optimizers(task, *model, tsampler, vsampler, *loss, cmd_criterion, table);
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
