#include "ncv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        using namespace ncv;

        ncv::init();

        const strings_t task_ids = task_manager_t::instance().ids();

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("task",
                boost::program_options::value<string_t>(),
                ("tasks to choose from: " + text::concatenate(task_ids, ", ")).c_str());
        po_desc.add_options()("task-dir",
                boost::program_options::value<string_t>(),
                "directory to load task data from");        
        po_desc.add_options()("save-dir",
                boost::program_options::value<string_t>(),
                "directory to save task samples to");
        po_desc.add_options()("save-group-rows",
                boost::program_options::value<size_t>()->default_value(32),
                "number of task samples to group in a row");
        po_desc.add_options()("save-group-cols",
                boost::program_options::value<size_t>()->default_value(32),
                "number of task samples to group in a column");
	
        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("task") ||
                !po_vm.count("task-dir") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_task = po_vm["task"].as<string_t>();
        const string_t cmd_task_dir = po_vm["task-dir"].as<string_t>();
        const string_t cmd_save_dir = po_vm.count("save-dir") ? po_vm["save-dir"].as<string_t>() : "";
        const size_t cmd_save_group_rows = math::clamp(po_vm["save-group-rows"].as<size_t>(), 1, 128);
        const size_t cmd_save_group_cols = math::clamp(po_vm["save-group-cols"].as<size_t>(), 1, 128);

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

        // save samples as images
        if (!cmd_save_dir.empty())
        {
                for (size_t f = 0; f < rtask->n_folds(); f ++)
                {
                        const fold_t train_fold = std::make_pair(f, protocol::train);
                        const fold_t test_fold = std::make_pair(f, protocol::test);

                        const string_t train_path = cmd_save_dir + "/" + cmd_task + "_train_fold" + text::to_string(f + 1);
                        const string_t test_path = cmd_save_dir + "/" + cmd_task + "_test_fold" + text::to_string(f + 1);

                        rtask->save(train_fold, train_path, cmd_save_group_rows, cmd_save_group_cols);
                        rtask->save(test_fold, test_path, cmd_save_group_rows, cmd_save_group_cols);
                }
        }
		
        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
