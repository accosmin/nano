#include "nano.h"
#include "task_util.h"
#include "text/cmdline.h"
#include "math/numeric.h"
#include "measure_and_log.h"
#include "text/concatenate.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        const strings_t task_ids = nano::get_tasks().ids();

        // parse the command line
        nano::cmdline_t cmdline("describe a task");
        cmdline.add("", "task",                 ("tasks to choose from: " + nano::concatenate(task_ids, ", ")).c_str());
        cmdline.add("", "task-params",          "task parameters (if any)", "dir=.");
        cmdline.add("", "save-dir",             "directory to save task samples to");
        cmdline.add("", "save-group-rows",      "number of task samples to group in a row", "32");
        cmdline.add("", "save-group-cols",      "number of task samples to group in a column", "32");

        cmdline.process(argc, argv);

        if (!cmdline.has("task"))
        {
                cmdline.usage();
        }

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_save_grows = nano::clamp(cmdline.get<tensor_size_t>("save-group-rows"), 1, 128);
        const auto cmd_save_gcols = nano::clamp(cmdline.get<tensor_size_t>("save-group-cols"), 1, 128);

        // create task
        const auto task = nano::get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        nano::measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        // describe task
        nano::describe(*task);

        // save samples as images
        if (cmdline.has("save-dir"))
        {
                const auto cmd_save_dir = cmdline.get<string_t>("save-dir");
                for (size_t f = 0; f < task->n_folds(); ++ f)
                {
                        for (auto p : {protocol::train, protocol::valid, protocol::test})
                        {
                                const auto fold = fold_t{f, p};
                                const auto path = cmd_save_dir + "/" + cmd_task + "_" + to_string(p) + to_string(f + 1);
                                nano::measure_and_log(
                                        [&] () { save_as_images(*task, fold, path, cmd_save_grows, cmd_save_gcols); },
                                        "save samples as images to <" + path + "*.png>");
                        }
                }
        }

        // OK
        nano::log_info() << nano::done;
        return EXIT_SUCCESS;
}
