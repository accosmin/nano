#include "nano.h"
#include "task_util.h"
#include "text/cmdline.h"
#include "math/numeric.h"
#include "measure_and_log.h"
#include "vision/image_grid.h"

using namespace nano;

static image_t make_image(const tensor3d_t& data)
{
        image_t image;
        image.from_tensor(data);
        image.make_rgba();
        return image;
}

static void save_as_images(const iterator_t& iterator, const task_t& task, const fold_t& fold, const string_t& basepath,
        const coord_t grows, const coord_t gcols,
        const size_t trials = 16,
        const coord_t border = 8,
        const rgba_t bkcolor = rgba_t{225, 225, 0, 255})
{
        const auto size = task.size(fold);
        const auto rows = std::get<1>(task.idims());
        const auto cols = std::get<2>(task.idims());

        image_grid_t grid(rows, cols, grows, gcols, border, bkcolor);

        // compose the image block with the original samples
        size_t i = 0;
        for (coord_t r = 0; r < grows; ++ r, ++ i)
        {
                for (coord_t c = 0; c < gcols && i < size; ++ c, ++ i)
                {
                        grid.set(r, c, make_image(task.input(fold, i)));
                }
        }

        grid.image().save(basepath + "_orig.png");

        // compose the image block with generated samples
        for (size_t t = 0; t < trials; ++ t)
        {
                i = 0;
                for (coord_t r = 0; r < grows; ++ r, ++ i)
                {
                        for (coord_t c = 0; c < gcols && i < size; ++ c, ++ i)
                        {
                                grid.set(r, c, make_image(iterator.input(task, fold, i)));
                        }
                }

                grid.image().save(basepath + "_trial" + to_string(t + 1) + ".png");
        }
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("describe the augmented training samples");
        cmdline.add("", "task",                 "[" + concatenate(get_tasks().ids()) + "]");
        cmdline.add("", "task-params",          "task parameters (if any)", "-");
        cmdline.add("", "iterator",             "[" + concatenate(get_iterators().ids()) + "]");
        cmdline.add("", "iterator-params",      "task iterator parameters (if any)", "-");
        cmdline.add("", "save-dir",             "directory to save samples to");
        cmdline.add("", "save-trials",          "number of sample generation trials", "16");
        cmdline.add("", "save-group-rows",      "number of samples to group in a row", "32");
        cmdline.add("", "save-group-cols",      "number of samples to group in a column", "32");

        cmdline.process(argc, argv);

        if (    !cmdline.has("task") ||
                !cmdline.has("iterator"))
        {
                cmdline.usage();
        }

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_iterator = cmdline.get<string_t>("iterator");
        const auto cmd_iterator_params = cmdline.get<string_t>("iterator-params");
        const auto cmd_save_trials = cmdline.get<size_t>("save-trials");
        const auto cmd_save_grows = clamp(cmdline.get<coord_t>("save-group-rows"), 1, 128);
        const auto cmd_save_gcols = clamp(cmdline.get<coord_t>("save-group-cols"), 1, 128);

        // create & load task
        const auto task = get_tasks().get(cmd_task, cmd_task_params);

        measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        describe(*task, cmd_task);

        // create iterator
        const auto iterator = get_iterators().get(cmd_iterator, cmd_iterator_params);

        // save samples as images
        if (cmdline.has("save-dir"))
        {
                const auto cmd_save_dir = cmdline.get<string_t>("save-dir");
                for (size_t f = 0; f < task->fsize(); ++ f)
                {
                        const auto fold = fold_t{f, protocol::train};
                        const auto path = cmd_save_dir + "/" + cmd_task + "_" + cmd_iterator + "_train" + to_string(f + 1);
                        measure_and_log(
                                [&] () { save_as_images(*iterator, *task, fold, path, cmd_save_grows, cmd_save_gcols, cmd_save_trials); },
                                "save samples as images to <" + path + "*.png>");
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
