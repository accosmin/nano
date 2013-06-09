#include "ncv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        const ncv::strings_t task_ids = ncv::task_manager_t::instance().ids();
        const ncv::strings_t loss_ids = ncv::loss_manager_t::instance().ids();

        const ncv::strings_t model_ids = ncv::model_manager_t::instance().ids();
        const ncv::strings_t model_descriptions = ncv::model_manager_t::instance().descriptions();

        ncv::string_t po_desc_models;
        po_desc_models += "models to choose from:";
        for (ncv::size_t i = 0; i < model_ids.size(); i ++)
        {
                po_desc_models += "\n  " +
                                ncv::text::resize(model_ids[i], 16) +
                                ncv::text::resize(model_descriptions[i], 32);
        }

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("task",
                boost::program_options::value<ncv::string_t>(),
                ("tasks to choose from: " + ncv::text::concatenate(task_ids, ", ")).c_str());
        po_desc.add_options()("task-dir",
                boost::program_options::value<ncv::string_t>(),
                "directory to load task data from");
        po_desc.add_options()("loss",
                boost::program_options::value<ncv::string_t>(),
                ("losses to choose from: " + ncv::text::concatenate(loss_ids, ", ")).c_str());
        po_desc.add_options()("model",
                boost::program_options::value<ncv::string_t>(),
                po_desc_models.c_str());
        po_desc.add_options()("model-params",
                boost::program_options::value<ncv::string_t>()->default_value(""),
                "model parameters (if any) as specified in the chosed model's description");
        po_desc.add_options()("trials",
                boost::program_options::value<ncv::size_t>(),
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

        const ncv::string_t cmd_task = po_vm["task"].as<ncv::string_t>();
        const ncv::string_t cmd_task_dir = po_vm["task-dir"].as<ncv::string_t>();
        const ncv::string_t cmd_loss = po_vm["loss"].as<ncv::string_t>();
        const ncv::string_t cmd_model = po_vm["model"].as<ncv::string_t>();
        const ncv::string_t cmd_model_params = po_vm["model-params"].as<ncv::string_t>();
        const ncv::size_t cmd_trials = po_vm["trials"].as<ncv::size_t>();
        ncv::timer_t timer;

        // create task
        ncv::rtask_t rtask = ncv::task_manager_t::instance().get(cmd_task, "");
        if (!rtask)
        {
                ncv::log_error() << "<<< failed to load task <" << cmd_task << ">!";
                return EXIT_FAILURE;
        }

        // load task data
        timer.start();
        if (!rtask->load(cmd_task_dir))
        {
                ncv::log_error() << "<<< failed to load task <" << cmd_task
                                 << "> from directory <" << cmd_task_dir << ">!";
                return EXIT_FAILURE;
        }
        else
        {
                ncv::log_info() << "<<< loaded task in " << timer.elapsed() << ".";
        }

        // describe task
        ncv::log_info() << "images: " << rtask->n_images() << ".";
        ncv::log_info() << "sample: #rows = " << rtask->n_rows()
                        << ", #cols = " << rtask->n_cols()
                        << ", #outputs = " << rtask->n_outputs()
                        << ", #folds = " << rtask->n_folds() << ".";

        for (ncv::size_t f = 0; f < rtask->n_folds(); f ++)
        {
                const ncv::fold_t train_fold = std::make_pair(f, ncv::protocol::train);
                const ncv::fold_t test_fold = std::make_pair(f, ncv::protocol::test);

                ncv::log_info() << "fold [" << (f + 1) << "/" << rtask->n_folds()
                                << "]: #train samples = " << rtask->samples(train_fold).size()
                                << ", #test samples = " << rtask->samples(test_fold).size() << ".";
        }

        // create loss
        ncv::rloss_t rloss = ncv::loss_manager_t::instance().get(cmd_loss, "");
        if (!rloss)
        {
                ncv::log_error() << "<<< failed to load loss <" << cmd_loss << ">!";
                return EXIT_FAILURE;
        }

        // create model
        ncv::rmodel_t rmodel = ncv::model_manager_t::instance().get(cmd_model, cmd_model_params);
        if (!rmodel)
        {
                ncv::log_error() << "<<< failed to load model <" << cmd_model << ">!";
                return EXIT_FAILURE;
        }

        // train & test models
        ncv::stats_t lstats, estats;
        for (ncv::size_t t = 0; t < cmd_trials; t ++)
        {
                for (ncv::size_t f = 0; f < rtask->n_folds(); f ++)
                {
                        const ncv::fold_t train_fold = std::make_pair(f, ncv::protocol::train);
                        const ncv::fold_t test_fold = std::make_pair(f, ncv::protocol::test);

                        timer.start();
                        if (!rmodel->train(*rtask, train_fold, *rloss))
                        {
                                ncv::log_error() << "<<< failed to train model <" << cmd_model << ">!";
                                break;
                        }
                        ncv::log_info() << "<<< training done in " << timer.elapsed() << ".";

                        timer.start();
                        ncv::scalar_t lvalue, lerror;
                        rmodel->test(*rtask, test_fold, *rloss, lvalue, lerror);
                        ncv::log_info() << "<<< test error: [" << lvalue << "/" << lerror
                                        << "] in " << timer.elapsed() << ".";

                        lstats.add(lvalue);
                        estats.add(lerror);
                }
        }

        // performance statistics
        ncv::log_info() << ">>> performance: loss value = " << lstats.avg() << " +/- " << lstats.stdev()
                        << " in [" << lstats.min() << ", " << lstats.max() << "].";
        ncv::log_info() << ">>> performance: loss error = " << estats.avg() << " +/- " << estats.stdev()
                        << " in [" << estats.min() << ", " << estats.max() << "].";

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
