#include "ncv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        const ncv::strings_t& task_names = ncv::task_manager::instance().names();
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("task,t",
                boost::program_options::value<ncv::string_t>(),
                ("task name (" + ncv::text::concatenate(task_names, ", ") + ")").c_str());
        po_desc.add_options()("dir,d",
                boost::program_options::value<ncv::string_t>(),
                "directory to load task data from");
	
        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("task") ||
                !po_vm.count("dir") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const ncv::string_t cmd_task = po_vm["task"].as<ncv::string_t>();
        const ncv::string_t cmd_dir = po_vm["dir"].as<ncv::string_t>();

        ncv::timer timer;

        // create task
        ncv::rtask rtask = ncv::task_manager::instance().get(cmd_task, "");
        if (!rtask)
        {
                ncv::log_error() << "<<< failed to load task <" << cmd_task << ">!";
                return EXIT_FAILURE;
        }

        // load task data
        timer.start();
        if (!rtask->load(cmd_dir))
        {
                ncv::log_error() << "<<< failed to load task <" << cmd_task
                                 << "> from directory <" << cmd_dir << ">!";
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
                        << ", #outputs = " << rtask->n_outputs() << ".";

        // TODO: describe folds: #training samples, #test samples, #samples with annotations
		
        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
