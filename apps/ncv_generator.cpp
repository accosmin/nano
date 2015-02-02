#include "libnanocv/nanocv.h"
#include "libnanocv/image_grid.h"
#include "libnanocv/util/measure.hpp"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        // prepare object string-based selection
        const strings_t model_ids = model_manager_t::instance().ids();

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "generate samples that produce the classes for a model");
        po_desc.add_options()("model",
                boost::program_options::value<string_t>(),
                text::concatenate(model_ids, ", ").c_str());
        po_desc.add_options()("model-file",
                boost::program_options::value<string_t>(),
                "filepath to load the model from");
        po_desc.add_options()("save-dir",
                boost::program_options::value<string_t>()->default_value("./"),
                "directory to save generated samples to");
        po_desc.add_options()("save-group-rows",
                boost::program_options::value<size_t>()->default_value(8),
                "number of generated samples to construct for each label [1, 128]");
        po_desc.add_options()("save-group-cols",
                boost::program_options::value<size_t>()->default_value(8),
                "number of generated samples to construct for each label [1, 128]");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("model") ||
                !po_vm.count("model-file") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_model = po_vm["model"].as<string_t>();
        const string_t cmd_input = po_vm["model-file"].as<string_t>();
        const string_t cmd_save_dir = po_vm.count("save-dir") ? po_vm["save-dir"].as<string_t>() : "";
        const size_t cmd_save_group_rows = math::clamp(po_vm["save-group-rows"].as<size_t>(), 1, 128);
        const size_t cmd_save_group_cols = math::clamp(po_vm["save-group-cols"].as<size_t>(), 1, 128);

        // create model
        const rmodel_t rmodel = model_manager_t::instance().get(cmd_model);

        // load model
        ncv::measure_critical_and_log(
                [&] () { return rmodel->load(cmd_input); },
                "loaded model",
                "failed to load model from <" + cmd_input + ">");

        // generate samples for each output class label
        const size_t labels = rmodel->osize();
        for (size_t l = 0; l < labels; l ++)
        {
                image_grid_t grid_image(rmodel->irows(), rmodel->icols(), cmd_save_group_rows, cmd_save_group_cols);

                const vector_t target = ncv::class_target(l, labels);
                for (size_t r = 0; r < cmd_save_group_rows; r ++)
                {
                        for (size_t c = 0; c < cmd_save_group_cols; c ++)
                        {
                                const tensor_t input = rmodel->generate(target);

                                image_t image;
                                if (!image.load(input))
                                {
                                        log_error() << "failed to map the generated input to RGBA image!";
                                        return EXIT_FAILURE;
                                }

                                grid_image.set(r, c, image);
                        }
                }

                const string_t path =
                        cmd_save_dir + "/" + boost::filesystem::basename(cmd_input) +
                        "_label" + text::to_string(l) + ".png";
                if (!grid_image.image().save(path))
                {
                        log_error() << "failed to save the generated input as RGBA image!";
                        return EXIT_FAILURE;
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
