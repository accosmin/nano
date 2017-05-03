#include <set>
#include "nano.h"
#include "task_util.h"
#include "text/cmdline.h"
#include "math/numeric.h"
#include "measure_and_log.h"
#include "text/concatenate.h"
#include "vision/image_grid.h"

using namespace nano;

static void save_as_images(const task_t& task, const fold_t& fold, const string_t& basepath,
        const coord_t grows, const coord_t gcols,
        const coord_t border = 8,
        const rgba_t bkcolor = rgba_t{225, 225, 0, 255})
{
        const auto size = task.size(fold);
        const auto rows = std::get<1>(task.idims());
        const auto cols = std::get<2>(task.idims());

        std::set<string_t> labels;
        for (size_t i = 0; i < size; ++ i)
        {
                labels.insert(task.label(fold, i));
        }

        // process each label separately
        for (const auto& label : labels)
        {
                for (size_t i = 0, g = 1; i < size; ++ g)
                {
                        image_grid_t grid_image(rows, cols, grows, gcols, border, bkcolor);

                        // compose the image block
                        for (coord_t r = 0; r < grows; ++ r)
                        {
                                for (coord_t c = 0; c < gcols && i < size; ++ c)
                                {
                                        for (; i < size && label != task.label(fold, i); ++ i) {}

                                        if (i < size)
                                        {
                                                image_t image;
                                                image.from_tensor(task.input(fold, i));
                                                image.make_rgba();
                                                grid_image.set(r, c, image);
                                                ++ i;
                                        }
                                }
                        }

                        // ... and save it
                        const auto path =
                                basepath +
                                (label.empty() ? "" : ("_" + label)) + "_group" + to_string(g) + ".png";
                        grid_image.image().save(path);
                }
        }
}

int main(int argc, const char *argv[])
{
        const auto task_ids = nano::get_tasks().ids();

        // parse the command line
        nano::cmdline_t cmdline("describe a task");
        cmdline.add("", "task",                 ("tasks to choose from: " + nano::concatenate(task_ids, ", ")).c_str());
        cmdline.add("", "task-params",          "task parameters (if any)", "-");
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
        const auto cmd_save_grows = nano::clamp(cmdline.get<coord_t>("save-group-rows"), 1, 128);
        const auto cmd_save_gcols = nano::clamp(cmdline.get<coord_t>("save-group-cols"), 1, 128);

        // create & load task
        const auto task = nano::get_tasks().get(cmd_task, cmd_task_params);

        nano::measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        nano::describe(*task, cmd_task);

        // save samples as images
        if (cmdline.has("save-dir"))
        {
                const auto cmd_save_dir = cmdline.get<string_t>("save-dir");
                for (size_t f = 0; f < task->fsize(); ++ f)
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
