#include "ncv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        const ncv::strings_t& task_names = ncv::task_manager_t::instance().names();
        const ncv::strings_t& loss_names = ncv::loss_manager_t::instance().names();
        const ncv::strings_t& model_names = ncv::model_manager_t::instance().names();

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("task",
                boost::program_options::value<ncv::string_t>(),
                ("task name (" + ncv::text::concatenate(task_names, ", ") + ")").c_str());
        po_desc.add_options()("task-dir",
                boost::program_options::value<ncv::string_t>(),
                "directory to load task data from");
        po_desc.add_options()("loss",
                boost::program_options::value<ncv::string_t>(),
                ("loss name (" + ncv::text::concatenate(loss_names, ", ") + ")").c_str());
        po_desc.add_options()("model",
                boost::program_options::value<ncv::string_t>(),
                ("model name (" + ncv::text::concatenate(model_names, ", ") + ")").c_str());
        po_desc.add_options()("trials",
                boost::program_options::value<ncv::size_t>(),
                "number of models to train & evaluate");
        po_desc.add_options()("iters",
                boost::program_options::value<ncv::size_t>()->default_value(128),
                "number of iterations [8, 4096]");
        po_desc.add_options()("eps",
                boost::program_options::value<ncv::scalar_t>()->default_value(1e-4),
                "convergence accuracy [1e-6, 1e-1]");
	
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
        const ncv::size_t cmd_trials = po_vm["trials"].as<ncv::size_t>();
        const ncv::size_t cmd_iters = ncv::math::clamp(po_vm["iters"].as<ncv::size_t>(), 8, 4096);
        const ncv::scalar_t cmd_eps = ncv::math::clamp(po_vm["eps"].as<ncv::scalar_t>(), 1e-6, 1e-1);

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
                ncv::log_info() << "<<< loaded task in " << timer.elapsed_string() << ".";
        }

        // describe task
        ncv::log_info() << "images: " << rtask->n_images() << ".";
        ncv::log_info() << "sample: #rows = " << rtask->n_rows()
                        << ", #cols = " << rtask->n_cols()
                        << ", #inputs = " << rtask->n_inputs()
                        << ", #outputs = " << rtask->n_outputs()
                        << ", #folds = " << rtask->n_folds() << ".";

        for (ncv::size_t f = 0; f < rtask->n_folds(); f ++)
        {
                const ncv::fold_t train_fold = std::make_pair(f, ncv::protocol::train);
                const ncv::fold_t test_fold = std::make_pair(f, ncv::protocol::test);

                ncv::log_info() << "fold [" << (f + 1) << "/" << rtask->n_folds()
                                << "]: #train samples = " << rtask->fold(train_fold).size()
                                << ", #test samples = " << rtask->fold(test_fold).size() << ".";
        }

        // create loss
        ncv::rloss_t rloss = ncv::loss_manager_t::instance().get(cmd_loss, "");
        if (!rloss)
        {
                ncv::log_error() << "<<< failed to load loss <" << cmd_loss << ">!";
                return EXIT_FAILURE;
        }

        // create model
        ncv::rmodel_t rmodel = ncv::model_manager_t::instance().get(cmd_model, "");
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
                        if (!rmodel->train(*rtask, train_fold, *rloss, cmd_iters, cmd_eps))
                        {
                                ncv::log_error() << "<<< failed to train model <" << cmd_model << ">!";
                                break;
                        }
                        ncv::log_info() << "<<< model trained in " << timer.elapsed_string() << ".";

                        timer.start();
                        ncv::scalar_t lvalue, lerror;
                        rmodel->test(*rtask, test_fold, *rloss, lvalue, lerror);
                        ncv::log_info() << "<<< model tested [" << lvalue << "/" << lerror
                                        << "] in " << timer.elapsed_string() << ".";

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
