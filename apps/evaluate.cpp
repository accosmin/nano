#include "nano.h"
#include "task_util.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "measure_and_log.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // prepare object string-based selection
        const strings_t task_ids = nano::get_tasks().ids();
        const strings_t loss_ids = nano::get_losses().ids();
        const strings_t model_ids = nano::get_models().ids();

        // parse the command line
        nano::cmdline_t cmdline("evaluate a model");
        cmdline.add("", "task",                 nano::concatenate(task_ids));
        cmdline.add("", "task-params",          "task parameters (if any)", "dir=.");
        cmdline.add("", "task-fold",            "fold index to use for testing", "0");
        cmdline.add("", "loss",                 nano::concatenate(loss_ids));
        cmdline.add("", "model",                nano::concatenate(model_ids));
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
        const auto task = nano::get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        nano::measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        // describe task
        nano::describe(*task, cmd_task);

        // create loss
        const auto loss = nano::get_losses().get(cmd_loss);

        // create sampler
        const auto sampler = nano::get_samplers().get("none");

        // create model
        const auto model = nano::get_models().get(cmd_model);

        // load model
        nano::measure_critical_and_log(
                [&] () { return model->load(cmd_model_file); },
                "load model from <" + cmd_model_file + ">");

        // test model
        const auto tfold = fold_t{cmd_task_fold, protocol::test};

        accumulator_t lacc(*model, *loss, *task, *sampler);
        lacc.mode(accumulator_t::type::value);
        lacc.threads(cmd_threads);

        nano::measure_and_log(
                [&] () { lacc.clear(); lacc.update(tfold); },
                "evaluate model");

        log_info() << "test=" << lacc.vstats().avg() << "|" << lacc.estats().avg() << "+/-" << lacc.estats().var() << ".";

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
