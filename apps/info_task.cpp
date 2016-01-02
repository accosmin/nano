#include "text/cmdline.h"
#include "cortex/cortex.h"
#include "text/concatenate.hpp"
#include "cortex/util/measure_and_log.hpp"

int main(int argc, char *argv[])
{
        using namespace cortex;

        cortex::init();

        const strings_t task_ids = cortex::get_tasks().ids();

        // parse the command line
        text::cmdline_t cmdline("describe a task");
        cmdline.add("", "task",                 ("tasks to choose from: " + text::concatenate(task_ids, ", ")).c_str());
        cmdline.add("", "task-dir",             "directory to load task data from");
        cmdline.add("", "task-params",          "task parameters (if any)", "");
        cmdline.add("", "save-dir",             "directory to save task samples to", "");
        cmdline.add("", "save-group-rows",      "number of task samples to group in a row", "32");
        cmdline.add("", "save-group-cols",      "number of task samples to group in a column", "32");
	
        cmdline.process(argc, argv);
        		
        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_dir = cmdline.get<string_t>("task-dir");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_save_dir = cmdline.get<string_t>("save-dir");
        const auto cmd_save_group_rows = math::clamp(cmdline.get<coord_t>("save-group-rows"), 1, 128);
        const auto cmd_save_group_cols = math::clamp(cmdline.get<coord_t>("save-group-cols"), 1, 128);

        // create task
        const auto task = cortex::get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        cortex::measure_critical_and_log(
                [&] () { return task->load(cmd_task_dir); },
                "load task <" + cmd_task + "> from <" + cmd_task_dir + ">");

        // describe task
        task->describe();

        // save samples as images
        if (!cmd_save_dir.empty())
        {
                for (size_t f = 0; f < task->fsize(); ++ f)
                {
                        const fold_t train_fold = std::make_pair(f, protocol::train);
                        const fold_t test_fold = std::make_pair(f, protocol::test);

                        const string_t train_path = cmd_save_dir + "/" + cmd_task + "_train_fold" + text::to_string(f + 1);
                        const string_t test_path = cmd_save_dir + "/" + cmd_task + "_test_fold" + text::to_string(f + 1);

                        task->save_as_images(train_fold, train_path, cmd_save_group_rows, cmd_save_group_cols);
                        task->save_as_images(test_fold, test_path, cmd_save_group_rows, cmd_save_group_cols);
                }
        }
		
        // OK
        cortex::log_info() << cortex::done;
        return EXIT_SUCCESS;
}
