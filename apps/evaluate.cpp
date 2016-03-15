#include "text/cmdline.h"
#include "cortex/cortex.h"
#include "cortex/sampler.h"
#include "cortex/evaluate.h"
#include "text/concatenate.hpp"
#include "cortex/util/measure_and_log.hpp"

int main(int argc, char *argv[])
{
        nano::init();

        using namespace nano;

        // prepare object string-based selection
        const strings_t task_ids = nano::get_tasks().ids();
        const strings_t loss_ids = nano::get_losses().ids();
        const strings_t model_ids = nano::get_models().ids();

        // parse the command line
        nano::cmdline_t cmdline("evaluate a model");
        cmdline.add("", "task",                 nano::concatenate(task_ids));
        cmdline.add("", "task-dir",             "directory to load task data from");
        cmdline.add("", "task-params",          "task parameters (if any)");
        cmdline.add("", "loss",                 nano::concatenate(loss_ids));
        cmdline.add("", "model",                nano::concatenate(model_ids));
        cmdline.add("", "model-file",           "filepath to load the model from");
        cmdline.add("", "save-dir",             "directory to save classification results to");
        cmdline.add("", "save-group-rows",      "number of samples to group in a row", "32");
        cmdline.add("", "save-group-cols",      "number of samples to group in a column", "32");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_dir = cmdline.get<string_t>("task-dir");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_input = cmdline.get<string_t>("model-file");
        const auto cmd_save_dir = cmdline.get<string_t>("save-dir");
        const auto cmd_save_group_rows = nano::clamp(cmdline.get<coord_t>("save-group-rows"), 1, 128);
        const auto cmd_save_group_cols = nano::clamp(cmdline.get<coord_t>("save-group-cols"), 1, 128);

        // create task
        const auto task = nano::get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        nano::measure_critical_and_log(
                [&] () { return task->load(cmd_task_dir); },
                "load task <" + cmd_task + "> from <" + cmd_task_dir + ">");

        // describe task
        task->describe();

        // create loss
        const auto loss = nano::get_losses().get(cmd_loss);

        // create criterion
        const auto criterion = nano::get_criteria().get("avg");

        // create model
        const auto model = nano::get_models().get(cmd_model);

        // load model
        nano::measure_critical_and_log(
                [&] () { return model->load(cmd_input); },
                "load model from <" + cmd_input + ">");

        // test model
        nano::stats_t<scalar_t> lstats, estats;
        for (size_t f = 0; f < task->fsize(); ++ f)
        {
                const fold_t test_fold = std::make_pair(f, protocol::test);

		// error rate
                const nano::timer_t timer;
                scalar_t lvalue, lerror;
                nano::evaluate(*task, test_fold, *loss, *criterion, *model, lvalue, lerror);
                log_info() << "<<< test error: [" << lvalue << "/" << lerror << "] in " << timer.elapsed() << ".";

                lstats(lvalue);
                estats(lerror);

		// per-label error rates
                sampler_t sampler(task->samples());
                sampler.push(test_fold);
                sampler.push(annotation::annotated);

                const samples_t samples = sampler.get();

                // split test samples into correctly classified & miss-classified
                samples_t ok_samples;
                samples_t nk_samples;

                for (size_t s = 0; s < samples.size(); ++ s)
                {
                        const sample_t& sample = samples[s];
                        const image_t& image = task->image(sample.m_index);

                        const vector_t target = sample.m_target;
                        const vector_t output = model->output(image.to_tensor(sample.m_region)).vector();

                        const indices_t tclasses = loss->labels(target);
                        const indices_t oclasses = loss->labels(output);

                        const bool ok = tclasses.size() == oclasses.size() &&
                                        std::mismatch(tclasses.begin(), tclasses.end(),
                                                      oclasses.begin()).first == tclasses.end();

                        (ok ? ok_samples : nk_samples).push_back(sample);
                }

                log_info() << "miss-classified " << nk_samples.size() << "/" << (samples.size()) << " = "
                           << (static_cast<scalar_t>(nk_samples.size()) / static_cast<scalar_t>(samples.size())) << ".";

                // save classification results
                if (!cmd_save_dir.empty())
                {
                        const string_t basepath = cmd_save_dir + "/" + cmd_task + "_test_fold" + nano::to_string(f + 1);

                        const coord_t grows = cmd_save_group_rows;
                        const coord_t gcols = cmd_save_group_cols;

                        const rgba_t ok_bkcolor = color::make_rgba(0, 225, 0);
                        const rgba_t nk_bkcolor = color::make_rgba(225, 0, 0);

                        // further split them by label
                        const strings_t labels = task->labels();
                        for (const string_t& label : labels)
                        {
                                const string_t lbasepath = basepath + "_" + label;

                                const samples_t label_ok_samples = sampler_t(ok_samples).push(label).get();
                                const samples_t label_nk_samples = sampler_t(nk_samples).push(label).get();
                                const samples_t label_ll_samples = sampler_t(samples).push(label).get();

                                log_info() << "miss-classified " << label_nk_samples.size()
                                           << "/" << label_ll_samples.size() << " = "
                                           << (static_cast<scalar_t>(label_nk_samples.size()) /
                                               static_cast<scalar_t>(label_ll_samples.size()))
                                           << " [" << label << "] samples.";

                                task->save_as_images(label_ok_samples, lbasepath + "_ok", grows, gcols, 8, ok_bkcolor);
                                task->save_as_images(label_nk_samples, lbasepath + "_nk", grows, gcols, 8, nk_bkcolor);
                        }
                }
        }

        // performance statistics
        log_info() << ">>> performance: loss value = " << lstats.avg() << " +/- " << lstats.stdev()
                   << " in [" << lstats.min() << ", " << lstats.max() << "].";
        log_info() << ">>> performance: loss error = " << estats.avg() << " +/- " << estats.stdev()
                   << " in [" << estats.min() << ", " << estats.max() << "].";

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
