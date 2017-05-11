#include "nano.h"
#include "task_util.h"
#include "text/cmdline.h"
#include "text/filesystem.h"
#include "measure_and_log.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("train a model");
        cmdline.add("", "task",                 "[" + concatenate(get_tasks().ids()) + "]");
        cmdline.add("", "task-params",          "task parameters (if any)", "dir=.");
        cmdline.add("", "task-fold",            "fold index to use for training", "0");
        cmdline.add("", "model",                "[" + concatenate(get_models().ids()) + "]");
        cmdline.add("", "model-params",         "model parameters (if any)");
        cmdline.add("", "model-file",           "filepath to save the model to");
        cmdline.add("", "trainer",              "[" + concatenate(get_trainers().ids()) + "]");
        cmdline.add("", "trainer-params",       "trainer parameters (if any)");
        cmdline.add("", "loss",                 "[" + concatenate(get_losses().ids()) + "]");
        cmdline.add("", "iterator",             "[" + concatenate(get_iterators().ids()) + "]", "default");
        cmdline.add("", "iterator-params",      "task iterator parameters (if any)", "");
        cmdline.add("", "threads",              "number of threads to use", logical_cpus());

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_task_fold = cmdline.get<size_t>("task-fold");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_model_params = cmdline.get<string_t>("model-params");
        const auto cmd_model_file = cmdline.get<string_t>("model-file");
        const auto cmd_state_file = nano::dirname(cmd_model_file) + nano::stem(cmd_model_file) + ".state";
        const auto cmd_trainer = cmdline.get<string_t>("trainer");
        const auto cmd_trainer_params = cmdline.get<string_t>("trainer-params");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_iterator = cmdline.get<string_t>("iterator");
        const auto cmd_iterator_params = cmdline.get<string_t>("iterator-params");
        const auto cmd_threads = cmdline.get<size_t>("threads");

        // create task
        const auto task = nano::get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        nano::measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        // describe task
        nano::describe(*task, cmd_task);

        // create loss
        const auto loss = nano::get_losses().get(cmd_loss);

        // create iterator
        const auto iterator = nano::get_iterators().get(cmd_iterator, cmd_iterator_params);

        // create model
        const auto model = nano::get_models().get(cmd_model, cmd_model_params);
        model->configure(*task);
        model->random();
        model->describe();

        if (*model != *task)
        {
                log_error() << "mis-matching model and task!";
                return EXIT_FAILURE;
        }

        // create trainer
        const auto trainer = nano::get_trainers().get(cmd_trainer, cmd_trainer_params);

        // train model
        trainer_result_t result;
        nano::measure_critical_and_log([&] ()
                {
                        result = trainer->train(*iterator, *task, cmd_task_fold, cmd_threads, *loss, *model);
                        return result.valid();
                },
                "train model");

        // save the model & its optimization history
        nano::measure_critical_and_log(
                [&] () { return model->save(cmd_model_file); },
                "save model to <" + cmd_model_file + ">");

        nano::measure_critical_and_log(
                [&] () { return nano::save(cmd_state_file, result.optimum_states()); },
                "save state to <" + cmd_state_file + ">");

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
