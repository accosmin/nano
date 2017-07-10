#include "loss.h"
#include "task.h"
#include "model.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "measure_and_log.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("evaluate a model");
        cmdline.add("", "task",                 "[" + concatenate(get_tasks().ids()) + "]");
        cmdline.add("", "task-params",          "task parameters (if any)", "-");
        cmdline.add("", "task-fold",            "fold index to use for testing", "0");
        cmdline.add("", "loss",                 "[" + concatenate(get_losses().ids()) + "]");
        cmdline.add("", "model",                "[" + concatenate(get_models().ids()) + "]");
        cmdline.add("", "model-file",           "filepath to load the model from");
        cmdline.add("", "threads",              "number of threads to use (0 - all available)", "0");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_task_fold = cmdline.get<size_t>("task-fold");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_model_file = cmdline.get<string_t>("model-file");
        const auto cmd_threads = cmdline.get<size_t>("threads");

        // create task
        const auto task = get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        // describe task
        describe(*task, cmd_task);

        // create loss
        const auto loss = get_losses().get(cmd_loss);

        // create model
        const auto model = get_models().get(cmd_model);

        // load model
        measure_critical_and_log(
                [&] () { return model->load(cmd_model_file); },
                "load model from <" + cmd_model_file + ">");

        // test model
        accumulator_t lacc(*model, *loss);
        lacc.mode(accumulator_t::type::value);
        lacc.threads(cmd_threads);

        measure_and_log(
                [&] () { lacc.update(*task, fold_t{cmd_task_fold, protocol::test}); },
                "evaluate model");

        log_info() << "test=" << lacc.vstats().avg() << "|" << lacc.estats().avg() << "+/-" << lacc.estats().var() << ".";

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
