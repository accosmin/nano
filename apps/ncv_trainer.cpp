#include "nanocv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        // prepare object string-based selection
        const strings_t task_ids = task_manager_t::instance().ids();
        const strings_t loss_ids = loss_manager_t::instance().ids();

        const strings_t model_ids = model_manager_t::instance().ids();
        const strings_t model_descriptions = model_manager_t::instance().descriptions();

        const strings_t trainer_ids = trainer_manager_t::instance().ids();
        const strings_t trainer_descriptions = trainer_manager_t::instance().descriptions();

        string_t po_desc_models;
        for (size_t i = 0; i < model_ids.size(); i ++)
        {
                po_desc_models +=
                        "  " + text::resize(model_ids[i], 16) +
                        text::resize(model_descriptions[i], 32) + (i + 1 == model_ids.size() ? "" : "\n");
        }

        string_t po_desc_trainers;
        for (size_t i = 0; i < trainer_ids.size(); i ++)
        {
                po_desc_trainers +=
                        "  " + text::resize(trainer_ids[i], 16) +
                        text::resize(trainer_descriptions[i], 32) + (i + 1 == trainer_ids.size() ? "" : "\n");
        }

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("task",
                boost::program_options::value<string_t>(),
                text::concatenate(task_ids, ", ").c_str());
        po_desc.add_options()("task-dir",
                boost::program_options::value<string_t>(),
                "directory to load task data from");
        po_desc.add_options()("loss",
                boost::program_options::value<string_t>(),
                text::concatenate(loss_ids, ", ").c_str());
        po_desc.add_options()("model",
                boost::program_options::value<string_t>(),
                po_desc_models.c_str());
        po_desc.add_options()("model-params",
                boost::program_options::value<string_t>()->default_value(""),
                "model parameters (if any) as specified in the chosen model's description");
        po_desc.add_options()("trainer",
                boost::program_options::value<string_t>(),
                po_desc_trainers.c_str());
        po_desc.add_options()("trainer-params",
                boost::program_options::value<string_t>()->default_value(""),
                "trainer parameters (if any) as specified in the chosen trainer's description");
        po_desc.add_options()("threads",
                boost::program_options::value<size_t>()->default_value(0),
                "number of threads to use (0 - all available)");
        po_desc.add_options()("trials",
                boost::program_options::value<size_t>(),
                "number of models to train & evaluate");
        po_desc.add_options()("output",
                boost::program_options::value<string_t>()->default_value(""),
                "filepath to save the best model to");
	
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
                !po_vm.count("trials") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_task = po_vm["task"].as<string_t>();
        const string_t cmd_task_dir = po_vm["task-dir"].as<string_t>();
        const string_t cmd_loss = po_vm["loss"].as<string_t>();
        const string_t cmd_model = po_vm["model"].as<string_t>();
        const string_t cmd_model_params = po_vm["model-params"].as<string_t>();
        const string_t cmd_trainer = po_vm["trainer"].as<string_t>();
        const string_t cmd_trainer_params = po_vm["trainer-params"].as<string_t>();
        const size_t cmd_threads = po_vm["threads"].as<size_t>();
        const size_t cmd_trials = po_vm["trials"].as<size_t>();
        const string_t cmd_output = po_vm["output"].as<string_t>();

        // create task
        const rtask_t rtask = task_manager_t::instance().get(cmd_task);

        // load task data
        ncv::measure_critical_call(
                [&] () { return rtask->load(cmd_task_dir); },
                "task loaded",
                "failed to load task <" + cmd_task + "> from directory <" + cmd_task_dir + ">");

        // describe task
        log_info() << "images: " << rtask->n_images() << ".";
        log_info() << "sample: #rows = " << rtask->n_rows()
                   << ", #cols = " << rtask->n_cols()
                   << ", #outputs = " << rtask->n_outputs()
                   << ", #folds = " << rtask->n_folds() << ".";

        for (size_t f = 0; f < rtask->n_folds(); f ++)
        {
                sampler_t trsampler(*rtask), tesampler(*rtask);
                trsampler.setup(fold_t(f, protocol::train));
                tesampler.setup(fold_t(f, protocol::test));

                log_info() << "fold [" << (f + 1) << "/" << rtask->n_folds()
                           << "]: #train samples = " << trsampler.size()
                           << ", #test samples = " << tesampler.size() << ".";
        }

        // create loss
        const rloss_t rloss = loss_manager_t::instance().get(cmd_loss);

        // create model
        const rmodel_t rmodel = model_manager_t::instance().get(cmd_model, cmd_model_params);

        // create trainer
        const rtrainer_t rtrainer = trainer_manager_t::instance().get(cmd_trainer, cmd_trainer_params);

        // train & test models
        std::map<scalar_t, rmodel_t> models;

        stats_t<scalar_t> lstats, estats;
        for (size_t t = 0; t < cmd_trials; t ++)
        {
                for (size_t f = 0; f < rtask->n_folds(); f ++)
                {
                        const fold_t train_fold = std::make_pair(f, protocol::train);
                        const fold_t test_fold = std::make_pair(f, protocol::test);

                        // train
                        ncv::measure_critical_call(
                                [&] () { return rtrainer->train(*rtask, train_fold, *rloss, cmd_threads, *rmodel); },
                                "model trained",
                                "failed to train model");

                        // test
                        scalar_t lvalue, lerror;
                        ncv::measure_call(
                                [&] () { ncv::test(*rtask, test_fold, *rloss, *rmodel, lvalue, lerror); },
                                "model tested");
                        log_info() << "<<< test error: [" << lvalue << "/" << lerror << "].";

                        lstats(lvalue);
                        estats(lerror);

                        // update the best model
                        models[lerror] = rmodel->clone();
                }
        }        

        // performance statistics
        log_info() << ">>> performance: loss value = " << lstats.avg() << " +/- " << lstats.stdev()
                   << " in [" << lstats.min() << ", " << lstats.max() << "].";
        log_info() << ">>> performance: loss error = " << estats.avg() << " +/- " << estats.stdev()
                   << " in [" << estats.min() << ", " << estats.max() << "].";

        // save the best model (if any trained)
        if (!models.empty() && !cmd_output.empty())
        {
                ncv::measure_critical_call(
                        [&] () { return models.begin()->second->save(cmd_output); },
                        "saved model",
                        "failed to save model to <" + cmd_output + ">");
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
