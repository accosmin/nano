#include "ncv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

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
                        "\t" + text::resize(model_ids[i], 16) +
                        text::resize(model_descriptions[i], 32) + (i + 1 == model_ids.size() ? "" : "\n");
        }

        string_t po_desc_trainers;
        for (size_t i = 0; i < model_ids.size(); i ++)
        {
                po_desc_trainers +=
                        "\t" + text::resize(trainer_ids[i], 16) +
                        text::resize(trainer_descriptions[i], 32) + (i + 1 == model_ids.size() ? "" : "\n");
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
                "model parameters (if any) as specified in the chosed model's description");
        po_desc.add_options()("trainer",
                boost::program_options::value<string_t>(),
                po_desc_trainers.c_str());
        po_desc.add_options()("trainer-params",
                boost::program_options::value<string_t>()->default_value(""),
                "trainer parameters (if any) as specified in the chosed trainer's description");
        po_desc.add_options()("trials",
                boost::program_options::value<size_t>(),
                "number of models to train & evaluate");
	
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
        const size_t cmd_trials = po_vm["trials"].as<size_t>();
        ncv::timer_t timer;

        // create task
        rtask_t rtask = task_manager_t::instance().get(cmd_task);
        if (!rtask)
        {
                log_error() << "<<< failed to load task <" << cmd_task << ">!";
                return EXIT_FAILURE;
        }

        // load task data
        timer.start();
        if (!rtask->load(cmd_task_dir))
        {
                log_error() << "<<< failed to load task <" << cmd_task
                                 << "> from directory <" << cmd_task_dir << ">!";
                return EXIT_FAILURE;
        }
        else
        {
                log_info() << "<<< loaded task in " << timer.elapsed() << ".";
        }

        // describe task
        log_info() << "images: " << rtask->n_images() << ".";
        log_info() << "sample: #rows = " << rtask->n_rows()
                        << ", #cols = " << rtask->n_cols()
                        << ", #outputs = " << rtask->n_outputs()
                        << ", #folds = " << rtask->n_folds() << ".";

        for (size_t f = 0; f < rtask->n_folds(); f ++)
        {
                const fold_t train_fold = std::make_pair(f, protocol::train);
                const fold_t test_fold = std::make_pair(f, protocol::test);

                log_info() << "fold [" << (f + 1) << "/" << rtask->n_folds()
                                << "]: #train samples = " << rtask->samples(train_fold).size()
                                << ", #test samples = " << rtask->samples(test_fold).size() << ".";
        }

        // create loss
        rloss_t rloss = loss_manager_t::instance().get(cmd_loss);
        if (!rloss)
        {
                log_error() << "<<< failed to load loss <" << cmd_loss << ">!";
                return EXIT_FAILURE;
        }

        // create model
        rmodel_t rmodel = model_manager_t::instance().get(cmd_model, cmd_model_params);
        if (!rmodel)
        {
                log_error() << "<<< failed to load model <" << cmd_model << ">!";
                return EXIT_FAILURE;
        }

        // create trainer
        rtrainer_t rtrainer = trainer_manager_t::instance().get(cmd_trainer, cmd_trainer_params);
        if (!rtrainer)
        {
                log_error() << "<<< failed to load trainer <" << cmd_trainer << ">!";
                return EXIT_FAILURE;
        }

        // train & test models
        stats_t lstats, estats;
        for (size_t t = 0; t < cmd_trials; t ++)
        {
                for (size_t f = 0; f < rtask->n_folds(); f ++)
                {
                        const fold_t train_fold = std::make_pair(f, protocol::train);
                        const fold_t test_fold = std::make_pair(f, protocol::test);

                        timer.start();
                        if (!rtrainer->train(*rtask, train_fold, *rloss, *rmodel))
                        {
                                log_error() << "<<< failed to train model <" << cmd_model << ">!";
                                break;
                        }
                        log_info() << "<<< training done in " << timer.elapsed() << ".";

                        timer.start();
                        scalar_t lvalue, lerror;
                        ncv::test(*rmodel, *rtask, test_fold, *rloss, lvalue, lerror);
                        log_info() << "<<< test error: [" << lvalue << "/" << lerror
                                        << "] in " << timer.elapsed() << ".";

                        lstats.add(lvalue);
                        estats.add(lerror);
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
