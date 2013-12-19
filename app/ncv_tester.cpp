#include "ncv.h"
#include "util/stats.hpp"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        // prepare object string-based selection
        const strings_t task_ids = task_manager_t::instance().ids();
        const strings_t loss_ids = loss_manager_t::instance().ids();
        const strings_t model_ids = model_manager_t::instance().ids();

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
                text::concatenate(model_ids, ", ").c_str());
        po_desc.add_options()("input",
                boost::program_options::value<string_t>(),
                "filepath to load the model from");

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
                !po_vm.count("input") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_task = po_vm["task"].as<string_t>();
        const string_t cmd_task_dir = po_vm["task-dir"].as<string_t>();
        const string_t cmd_loss = po_vm["loss"].as<string_t>();
        const string_t cmd_model = po_vm["model"].as<string_t>();
        const string_t cmd_input = po_vm["input"].as<string_t>();

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
        rmodel_t rmodel = model_manager_t::instance().get(cmd_model);
        if (!rmodel)
        {
                log_error() << "<<< failed to load model <" << cmd_model << ">!";
                return EXIT_FAILURE;
        }

        // load best model
        timer.start();
        if (!rmodel->load(cmd_input))
        {
                log_error() << "<<< failed to load model from <" << cmd_input << ">!";
                return EXIT_FAILURE;
        }
        log_info() << "<<< model <" << cmd_input << "> loaded in " << timer.elapsed() << ".";

        // test models
        stats_t<scalar_t> lstats, estats;
        for (size_t f = 0; f < rtask->n_folds(); f ++)
        {
                const fold_t test_fold = std::make_pair(f, protocol::test);

                // test
                timer.start();
                scalar_t lvalue, lerror;
                ncv::test(*rtask, test_fold, *rloss, *rmodel, lvalue, lerror);
                log_info() << "<<< test error: ["
                           << lvalue << "/" << lerror << "] in " << timer.elapsed() << ".";

                lstats.add(lvalue);
                estats.add(lerror);
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
