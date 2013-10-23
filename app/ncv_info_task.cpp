#include "ncv.h"
#include "core/timer.h"
#include "core/logger.h"
#include "core/math/clamp.hpp"
#include <boost/program_options.hpp>

void save(const ncv::task_t& task, const ncv::fold_t& fold,
          const ncv::string_t& base_path,
          ncv::size_t group_rows, ncv::size_t group_cols)
{
        using namespace ncv;

        const size_t border = 8;
        const size_t rows = task.n_rows() * group_rows + border * (group_rows + 1);
        const size_t cols = task.n_cols() * group_cols + border * (group_cols + 1);

        rgba_matrix_t rgba(rows, cols);

        // process all samples ...
        const samples_t& samples = task.samples(fold);
        for (size_t i = 0, g = 1; i < samples.size(); i += group_rows * group_cols, g ++)
        {
                rgba.setConstant(color::make_rgba(225, 225, 0));

                // ... compose the image block
                for (size_t k = i, r = 0; r < group_rows; r ++)
                {
                        for (size_t c = 0; c < group_cols; c ++, k ++)
                        {
                                if (k < samples.size())
                                {
                                        const ncv::sample_t& sample = samples[k];
                                        const ncv::image_t& image = task.image(sample.m_index);
                                        const ncv::rect_t& region = sample.m_region;

                                        rgba.block(             task.n_rows() * r + border * (r + 1),
                                                                task.n_cols() * c + border * (c + 1),
                                                                task.n_rows(),
                                                                task.n_cols()) =
                                        image.m_rgba.block(
                                                                geom::top(region),
                                                                geom::left(region),
                                                                geom::height(region),
                                                                geom::width(region));
                                }
                        }
                }

                // ... and save it
                const string_t path = base_path + "_group" + text::to_string(g) + ".png";
                log_info() << "saving images to <" << path << "> ...";
                ncv::save_rgba(path, rgba);
        }
}

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

                        const string_t train_path = cmd_save_dir + "/" + cmd_task + "_train" + text::to_string(f + 1);
                        const string_t test_path = cmd_save_dir + "/" + cmd_task + "_test" + text::to_string(f + 1);

                        save(*rtask, train_fold, train_path, cmd_save_group_rows, cmd_save_group_cols);
                        save(*rtask, test_fold, test_path, cmd_save_group_rows, cmd_save_group_cols);
                }
        }
		
        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
