#include "libnanocv/nanocv.h"
#include "libnanocv/tasks/task_synthetic_shapes.h"
#include "libnanocv/util/thread.h"
#include "libnanocv/util/measure.hpp"
#include "libnanocv/util/tabulator.h"
#include "libnanocv/trainers/batch.h"
#include "libnanocv/trainers/minibatch.h"
#include "libnanocv/trainers/stochastic.h"

using namespace ncv;

template
<
        typename ttrainer
>
void test_optimizer(model_t& model, ttrainer trainer, const string_t& name, tabulator_t& table)
{
        const size_t cmd_trials = 16;

        stats_t<scalar_t> tvalues;
        stats_t<scalar_t> vvalues;
        stats_t<scalar_t> terrors;
        stats_t<scalar_t> verrors;

        const size_t usec = ncv::measure_robustly_usec([&] ()
        {
                model.random_params();

                const trainer_result_t result = trainer();

                tvalues(result.m_opt_state.m_tvalue);
                vvalues(result.m_opt_state.m_vvalue);

                terrors(result.m_opt_state.m_terror_avg);
                verrors(result.m_opt_state.m_verror_avg);
        }, cmd_trials);

        table.append(name)
                << tvalues.avg() << terrors.avg() << vvalues.avg() << verrors.avg()
                << (usec / 1000);
}

void test_optimizers(
        const task_t& task, model_t& model, const sampler_t& tsampler, const sampler_t& vsampler,
        const loss_t& loss, const string_t& criterion,
        const string_t& config_name)
{
        const size_t cmd_iterations = 32;
        const size_t cmd_epochs = cmd_iterations;
        const scalar_t cmd_epsilon = 1e-4;
        const bool verbose = false;

        // batch optimizers
        const auto batch_optimizers =
        {
                batch_optimizer::GD,
                batch_optimizer::CGD,
                batch_optimizer::LBFGS
        };

        // minibatch optimizers
        const auto minibatch_optimizers =
        {
                batch_optimizer::GD,
                batch_optimizer::CGD,
                batch_optimizer::LBFGS
        };

        // stochastic optimizers
        const auto stochastic_optimizers =
        {
                stochastic_optimizer::SG,
                stochastic_optimizer::SGA,
                stochastic_optimizer::SIA,
                stochastic_optimizer::AG,
                stochastic_optimizer::ADAGRAD,
                stochastic_optimizer::ADADELTA
        };

        // run optimizers and collect results
        tabulator_t table("optimizer\\");
        table.header() << "train loss" << "train error" << "valid loss" << "valid error" << "time [msec]";

        for (batch_optimizer optimizer : batch_optimizers)
        {
                test_optimizer(model, [&] ()
                {
                        return ncv::batch_train(
                                model, task, tsampler, vsampler, ncv::n_threads(),
                                loss, criterion, optimizer, cmd_iterations, cmd_epsilon, verbose);
                }, "batch [" + text::to_string(optimizer) + "]", table);
        }

        for (batch_optimizer optimizer : minibatch_optimizers)
        {
                test_optimizer(model, [&] ()
                {
                        return ncv::minibatch_train(
                                model, task, tsampler, vsampler, ncv::n_threads(),
                                loss, criterion, optimizer, cmd_epochs, cmd_epsilon, verbose);
                }, "minibatch [" + text::to_string(optimizer) + "]", table);
        }

        for (stochastic_optimizer optimizer : stochastic_optimizers)
        {
                test_optimizer(model, [&] ()
                {
                        return ncv::stochastic_train(
                                model, task, tsampler, vsampler, ncv::n_threads(),
                                loss, criterion, optimizer, cmd_epochs, verbose);
                }, "stochastic [" + text::to_string(optimizer) + "]", table);
        }

        table.print(std::cout);
}

int main(int argc, char *argv[])
{
        ncv::init();

        const size_t cmd_samples = 8 * 1024;
        const size_t cmd_rows = 16;
        const size_t cmd_cols = 16;
        const size_t cmd_outputs = 9;

        // create task
        synthetic_shapes_task_t task(
                "rows=" + text::to_string(cmd_rows) + "," +
                "cols=" + text::to_string(cmd_cols) + "," +
                "color=luma" + "," +
                "dims=" + text::to_string(cmd_outputs) + "," +
                "size=" + text::to_string(cmd_samples));
        task.load("");
	task.describe();

        // create training & validation samples
        sampler_t tsampler(task);
        tsampler.setup(sampler_t::atype::annotated);

        sampler_t vsampler(task);
        tsampler.split(80, vsampler);

        // construct models
        const string_t lmodel0;
        const string_t lmodel1 = lmodel0 + "linear:dims=16;act-snorm;";
        const string_t lmodel2 = lmodel1 + "linear:dims=16;act-snorm;";
        const string_t lmodel3 = lmodel2 + "linear:dims=16;act-snorm;";

        string_t cmodel;
        cmodel = cmodel + "conv:dims=8,rows=5,cols=5;pool-max;act-snorm;";
        cmodel = cmodel + "conv:dims=8,rows=3,cols=3;act-snorm;";

        const string_t outlayer = "linear:dims=" + text::to_string(cmd_outputs) + ";";

        strings_t cmd_networks =
        {
                lmodel0 + outlayer,
                lmodel1 + outlayer,
                lmodel2 + outlayer,
                lmodel3 + outlayer,

                cmodel + outlayer
        };

        const strings_t cmd_losses = { "classnll", "logistic" }; //loss_manager_t::instance().ids();
        const strings_t cmd_criteria = { "avg" }; //criterion_manager_t::instance().ids();

        // vary the model
        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                const rmodel_t model = model_manager_t::instance().get("forward-network", cmd_network);
                assert(model);
                model->resize(task, true);

                // vary the loss
                for (const string_t& cmd_loss : cmd_losses)
                {
                        log_info() << "<<< running loss [" << cmd_loss << "] ...";

                        const rloss_t loss = loss_manager_t::instance().get(cmd_loss);
                        assert(loss);

                        // vary the criteria
                        for (const string_t& cmd_criterion : cmd_criteria)
                        {
                                log_info() << "<<< running criterion [" << cmd_criterion << "] ...";

                                test_optimizers(task, *model, tsampler, vsampler, *loss, cmd_criterion,
                                                "loss [" + cmd_loss + "], criterion [" + cmd_criterion + "]");
                        }
                }

                log_info();
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
