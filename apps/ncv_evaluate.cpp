#include "libnanocv/nanocv.h"
#include "libnanocv/measure.hpp"
#include "libnanocv/evaluate.h"
#include "libnanocv/sampler.h"
#include <boost/program_options.hpp>
#include <algorithm>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        // prepare object string-based selection
        const strings_t task_ids = ncv::get_tasks().ids();
        const strings_t loss_ids = ncv::get_losses().ids();
        const strings_t model_ids = ncv::get_models().ids();

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("task",
                boost::program_options::value<string_t>(),
                text::concatenate(task_ids, ", ").c_str());
        po_desc.add_options()("task-dir",
                boost::program_options::value<string_t>(),
                "directory to load task data from");
        po_desc.add_options()("task-params",
                boost::program_options::value<string_t>()->default_value(""),
                "task parameters (if any)");
        po_desc.add_options()("loss",
                boost::program_options::value<string_t>(),
                text::concatenate(loss_ids, ", ").c_str());
        po_desc.add_options()("model",
                boost::program_options::value<string_t>(),
                text::concatenate(model_ids, ", ").c_str());
        po_desc.add_options()("model-file",
                boost::program_options::value<string_t>(),
                "filepath to load the model from");
        po_desc.add_options()("save-dir",
                boost::program_options::value<string_t>(),
                "directory to save classification results to");
        po_desc.add_options()("save-group-rows",
                boost::program_options::value<size_t>()->default_value(32),
                "number of samples to group in a row");
        po_desc.add_options()("save-group-cols",
                boost::program_options::value<size_t>()->default_value(32),
                "number of samples to group in a column");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("task") ||
                !po_vm.count("task-dir") ||
                !po_vm.count("loss") ||
                !po_vm.count("model") ||
                !po_vm.count("model-file") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_task = po_vm["task"].as<string_t>();
        const string_t cmd_task_dir = po_vm["task-dir"].as<string_t>();
        const string_t cmd_task_params = po_vm["task-params"].as<string_t>();
        const string_t cmd_loss = po_vm["loss"].as<string_t>();
        const string_t cmd_model = po_vm["model"].as<string_t>();
        const string_t cmd_input = po_vm["model-file"].as<string_t>();
        const string_t cmd_save_dir = po_vm.count("save-dir") ? po_vm["save-dir"].as<string_t>() : "";
        const size_t cmd_save_group_rows = math::clamp(po_vm["save-group-rows"].as<size_t>(), 1, 128);
        const size_t cmd_save_group_cols = math::clamp(po_vm["save-group-cols"].as<size_t>(), 1, 128);

        // create task
        const rtask_t rtask = ncv::get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        ncv::measure_critical_and_log(
                [&] () { return rtask->load(cmd_task_dir); },
                "loaded task",
                "failed to load task <" + cmd_task + "> from directory <" + cmd_task_dir + ">");

        // describe task
        rtask->describe();

        // create loss
        const rloss_t rloss = ncv::get_losses().get(cmd_loss);

        // create model
        const rmodel_t rmodel = ncv::get_models().get(cmd_model);

        // load model
        ncv::measure_critical_and_log(
                [&] () { return rmodel->load(cmd_input); },
                "loaded model",
                "failed to load model from <" + cmd_input + ">");

        // test model
        stats_t<scalar_t> lstats, estats;
        for (size_t f = 0; f < rtask->fsize(); f ++)
        {
                const fold_t test_fold = std::make_pair(f, protocol::test);

		// error rate
                const ncv::timer_t timer;
                scalar_t lvalue, lerror;
                ncv::evaluate(*rtask, test_fold, *rloss, *rmodel, lvalue, lerror);
                log_info() << "<<< test error: [" << lvalue << "/" << lerror << "] in " << timer.elapsed() << ".";

                lstats(lvalue);
                estats(lerror);

		// per-label error rates
		sampler_t sampler(*rtask);
                sampler.setup(test_fold);
                sampler.setup(sampler_t::atype::annotated);
                sampler.setup(sampler_t::stype::batch);

                const samples_t samples = sampler.get();

                // split test samples into correctly classified & miss-classified
                samples_t ok_samples;
                samples_t nk_samples;

                for (size_t s = 0; s < samples.size(); s ++)
                {
                        const sample_t& sample = samples[s];
                        const image_t& image = rtask->image(sample.m_index);

                        const vector_t target = sample.m_target;
                        const vector_t output = rmodel->output(image, sample.m_region).vector();

                        const indices_t tclasses = rloss->labels(target);
                        const indices_t oclasses = rloss->labels(output);

                        const bool ok = tclasses.size() == oclasses.size() &&
                                        std::mismatch(tclasses.begin(), tclasses.end(),
                                                      oclasses.begin()).first == tclasses.end();

                        (ok ? ok_samples : nk_samples).push_back(sample);
                }

                log_info() << "miss-classified " << nk_samples.size() << "/" << (samples.size()) 
                           << " = " << ((0.0 + nk_samples.size()) / (0.0 + samples.size())) << ".";

                // save classification results
                if (!cmd_save_dir.empty())
                {
                        const string_t basepath = cmd_save_dir + "/" + cmd_task + "_test_fold" + text::to_string(f + 1);

                        const size_t grows = cmd_save_group_rows;
                        const size_t gcols = cmd_save_group_cols;

                        const rgba_t ok_bkcolor = color::make_rgba(0, 225, 0);
                        const rgba_t nk_bkcolor = color::make_rgba(225, 0, 0);

                        // further split them by label
                        const strings_t labels = rtask->labels();
                        for (const string_t& label : labels)
                        {
                                const string_t lbasepath = basepath + "_" + label;

                                sampler_t ok_sampler(ok_samples); ok_sampler.setup(label);
                                sampler_t nk_sampler(nk_samples); nk_sampler.setup(label);
                                sampler_t ll_sampler(samples); ll_sampler.setup(label);

                                const samples_t label_ok_samples = ok_sampler.get();
                                const samples_t label_nk_samples = nk_sampler.get();
                                const samples_t label_ll_samples = ll_sampler.get();

                                log_info() << "miss-classified " << label_nk_samples.size()
                                           << "/" << label_ll_samples.size()
                                           << " = " << ((0.0 + label_nk_samples.size()) / (0.0 + label_ll_samples.size()))
                                           << " [" << label << "] samples.";

                                rtask->save_as_images(label_ok_samples, lbasepath + "_ok", grows, gcols, 8, ok_bkcolor);
                                rtask->save_as_images(label_nk_samples, lbasepath + "_nk", grows, gcols, 8, nk_bkcolor);
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
