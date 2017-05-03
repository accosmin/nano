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

static void save_as_images(const task_t& task, const fold_t& fold, sampler_t& sampler, const string_t& basepath,
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
                                tensor3d_t data = task.input(fold, i);
                                sampler.get(data, nullptr, nullptr);
                                grid.set(r, c, make_image(data));
                        }
                }

                grid.image().save(basepath + "_trial" + to_string(t + 1) + ".png");
        }
}

int main(int argc, const char *argv[])
{
        const auto task_ids = nano::get_tasks().ids();
        const auto sampler_ids = nano::get_samplers().ids();

        // parse the command line
        nano::cmdline_t cmdline("describe the augmented training samples");
        cmdline.add("", "task",                 ("tasks to choose from: " + nano::concatenate(task_ids, ", ")).c_str());
        cmdline.add("", "task-params",          "task parameters (if any)", "-");
        cmdline.add("", "sampler",              ("samplers to choose from: " + nano::concatenate(sampler_ids, ", ")).c_str());
        cmdline.add("", "sampler-params",       "sampler parameters (if any)", "-");
        cmdline.add("", "save-dir",             "directory to save samples to");
        cmdline.add("", "save-trials",          "number of sample generation trials", "16");
        cmdline.add("", "save-group-rows",      "number of samples to group in a row", "32");
        cmdline.add("", "save-group-cols",      "number of samples to group in a column", "32");

        cmdline.process(argc, argv);

        if (    !cmdline.has("task") ||
                !cmdline.has("sampler"))
        {
                cmdline.usage();
        }

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_sampler = cmdline.get<string_t>("sampler");
        const auto cmd_sampler_params = cmdline.get<string_t>("sampler-params");
        const auto cmd_save_trials = cmdline.get<size_t>("save-trials");
        const auto cmd_save_grows = nano::clamp(cmdline.get<coord_t>("save-group-rows"), 1, 128);
        const auto cmd_save_gcols = nano::clamp(cmdline.get<coord_t>("save-group-cols"), 1, 128);

        // create & load task
        const auto task = nano::get_tasks().get(cmd_task, cmd_task_params);

        nano::measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        nano::describe(*task, cmd_task);

        // create sampler
        const auto sampler = nano::get_samplers().get(cmd_sampler, cmd_sampler_params);

        // save samples as images
        if (cmdline.has("save-dir"))
        {
                const auto cmd_save_dir = cmdline.get<string_t>("save-dir");
                for (size_t f = 0; f < task->fsize(); ++ f)
                {
                        const auto fold = fold_t{f, protocol::train};
                        const auto path = cmd_save_dir + "/" + cmd_task + "_" + cmd_sampler + "_train" + to_string(f + 1);
                        nano::measure_and_log(
                                [&] () { save_as_images(*task, fold, *sampler, path, cmd_save_grows, cmd_save_gcols, cmd_save_trials); },
                                "save samples as images to <" + path + "*.png>");
                }
        }

        // OK
        nano::log_info() << nano::done;
        return EXIT_SUCCESS;
}
