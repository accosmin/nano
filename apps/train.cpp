#include "text/cmdline.h"
#include "text/align.hpp"
#include "cortex/cortex.h"
#include "cortex/evaluate.h"
#include "text/filesystem.h"
#include "text/concatenate.hpp"
#include "cortex/util/measure_and_log.hpp"

int main(int argc, char *argv[])
{
        using namespace cortex;

        cortex::init();

        // prepare object string-based selection
        const strings_t task_ids = cortex::get_tasks().ids();
        const strings_t loss_ids = cortex::get_losses().ids();
        const strings_t model_ids = cortex::get_models().ids();
        const strings_t trainer_ids = cortex::get_trainers().ids();
        const strings_t criterion_ids = cortex::get_criteria().ids();

        // parse the command line
        text::cmdline_t cmdline("train a model");
        cmdline.add("", "task",                 text::concatenate(task_ids));
        cmdline.add("", "task-dir",             "directory to load task data from");
        cmdline.add("", "task-params",          "task parameters (if any)", "<>");
        cmdline.add("", "loss",                 text::concatenate(loss_ids));
        cmdline.add("", "model",                text::concatenate(model_ids));
        cmdline.add("", "model-params",         "model parameters (if any)");
        cmdline.add("", "trainer",              text::concatenate(trainer_ids));
        cmdline.add("", "trainer-params",       "trainer parameters (if any)");
        cmdline.add("", "criterion",            text::concatenate(criterion_ids));
        cmdline.add("", "threads",              "number of threads to use (0 - all available)", "0");
        cmdline.add("", "trials",               "number of models to train & evaluate");
        cmdline.add("", "output",               "filepath to save the best model to");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_dir = cmdline.get<string_t>("task-dir");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_model_params = cmdline.get<string_t>("model-params");
        const auto cmd_trainer = cmdline.get<string_t>("trainer");
        const auto cmd_trainer_params = cmdline.get<string_t>("trainer-params");
        const auto cmd_criterion = cmdline.get<string_t>("criterion");
        const auto cmd_threads = cmdline.get<size_t>("threads");
        const auto cmd_trials = cmdline.get<size_t>("trials");
        const auto cmd_output = cmdline.get<string_t>("output");

        // create task
        const auto task = cortex::get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        cortex::measure_critical_and_log(
                [&] () { return task->load(cmd_task_dir); },
                "load task <" + cmd_task + "> from <" + cmd_task_dir + ">");

        // describe task
        task->describe();

        // create loss
        const auto loss = cortex::get_losses().get(cmd_loss);

        // create criterion
        const auto criterion = cortex::get_criteria().get(cmd_criterion);

        // create model
        const auto rmodel = cortex::get_models().get(cmd_model, cmd_model_params);

        // create trainer
        const auto trainer = cortex::get_trainers().get(cmd_trainer, cmd_trainer_params);

        // train & test models
        std::map<scalar_t, std::tuple<rmodel_t, trainer_states_t>> models;

        math::stats_t<scalar_t> lstats, estats;
        for (size_t t = 0; t < cmd_trials; ++ t)
        {
                for (size_t f = 0; f < task->fsize(); ++ f)
                {
                        const fold_t train_fold = std::make_pair(f, protocol::train);
                        const fold_t test_fold = std::make_pair(f, protocol::test);

                        // train
                        trainer_result_t result;
                        cortex::measure_critical_and_log([&] ()
                        {
                                result = trainer->train(*task, train_fold, *loss, cmd_threads, *criterion, *rmodel);
                                return result.valid();
                        },
                        "train model");

                        // test
                        scalar_t lvalue, lerror;
                        cortex::measure_and_log(
                                [&] () { cortex::evaluate(*task, test_fold, *loss, *criterion, *rmodel, lvalue, lerror); },
                                "test model");
                        log_info() << "<<< test error: [" << lvalue << "/" << lerror << "].";

                        lstats(lvalue);
                        estats(lerror);

                        // update the best model
                        models[lerror] = std::make_tuple(rmodel->clone(), result.optimum_states());
                }
        }

        // performance statistics
        log_info() << ">>> performance: loss value = " << lstats.avg() << " +/- " << lstats.stdev()
                   << " in [" << lstats.min() << ", " << lstats.max() << "].";
        log_info() << ">>> performance: loss error = " << estats.avg() << " +/- " << estats.stdev()
                   << " in [" << estats.min() << ", " << estats.max() << "].";

        // save the best model & optimization history (if any trained)
        if (!models.empty() && !cmd_output.empty())
        {
                const auto& opt_model = std::get<0>(models.begin()->second);
                const trainer_states_t& opt_states = std::get<1>(models.begin()->second);

                cortex::measure_critical_and_log(
                        [&] () { return opt_model->save(cmd_output); },
                        "save model to <" + cmd_output + ">");

                const string_t path = text::dirname(cmd_output) + text::stem(cmd_output) + ".state";

                cortex::measure_critical_and_log(
                        [&] () { return cortex::save(path, opt_states); },
                        "save state to <" + path + ">");
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
