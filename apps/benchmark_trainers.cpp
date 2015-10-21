#include "cortex/batch.h"
#include "cortex/table.h"
#include "cortex/cortex.h"
#include "thread/thread.h"
#include "cortex/measure.hpp"
#include "cortex/minibatch.h"
#include "cortex/stochastic.h"
#include "cortex/accumulator.h"
#include "text/concatenate.hpp"
#include "cortex/tasks/task_charset.h"

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
                ttrainer trainer)
        {
                math::stats_t<scalar_t> terrors;
                math::stats_t<scalar_t> verrors;

                log_info() << "<<< running " << name << " ...";

                const cortex::timer_t timer;

                for (const vector_t& x0 : x0s)
                {
                        model.load_params(x0);

                        const trainer_result_t result = trainer();
                        const trainer_state_t state = result.optimum_state();

                        terrors(state.m_terror_avg);
                        verrors(state.m_verror_avg);

                        log_info() << "<<< " << name << ", optimum config = {" << text::concatenate(result.optimum_config())
                                   << "}, optimum epoch = " << result.optimum_epoch()
                                   << ", error " << state.m_terror_avg << "/" << state.m_verror_avg << ".";
                }

                table.append(name)
                        << stats_to_string(terrors)
                        << stats_to_string(verrors)
                        << (timer.seconds());
        }

        void test_optimizers(
                const task_t& task, model_t& model, const sampler_t& tsampler, const sampler_t& vsampler,
                const loss_t& loss, const string_t& criterion, table_t& table)
        {
                const size_t cmd_trials = 16;

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

int main(int, char* [])
{
        cortex::init();

        const size_t cmd_rows = 16;
        const size_t cmd_cols = 16;
        const size_t cmd_samples = thread::n_threads() * 16 * 10;
        const color_mode cmd_color = color_mode::rgba;

        // create task
        charset_task_t task(charset::numeric, cmd_rows, cmd_cols, cmd_color, cmd_samples);
        task.load("");
	task.describe();

        const auto cmd_outputs = task.osize();

        // create training & validation samples
        sampler_t tsampler(task.samples());
        tsampler.setup(sampler_t::atype::annotated);

        sampler_t vsampler(task.samples());
        tsampler.split(80, vsampler);

        // construct models
//        const string_t lmodel0;
//        const string_t lmodel1 = lmodel0 + "linear:dims=64;act-snorm;";
//        const string_t lmodel2 = lmodel1 + "linear:dims=32;act-snorm;";
//        const string_t lmodel3 = lmodel2 + "linear:dims=16;act-snorm;";

        string_t cmodel;
        cmodel = cmodel + "conv:dims=16,rows=5,cols=5;pool-max;act-snorm;";
        cmodel = cmodel + "conv:dims=32,rows=3,cols=3;act-snorm;";

        const string_t outlayer = "linear:dims=" + text::to_string(cmd_outputs) + ";";

        strings_t cmd_networks =
        {
//                lmodel0 + outlayer,
//                lmodel1 + outlayer,
//                lmodel2 + outlayer,
//                lmodel3 + outlayer,

                cmodel + outlayer
        };

        const strings_t cmd_losses = { "classnll" };    //cortex::get_losses().ids();
        const strings_t cmd_criteria = { "avg" }; //cortex::get_criteria().ids();

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

                        table_t table("optimizer\\");
                        table.header() << "train error"
                                       << "valid error"
                                       << "time [msec]";

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
