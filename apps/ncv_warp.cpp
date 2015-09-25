#include "core/warp.h"
#include "core/image.h"
#include "core/string.h"
#include "core/measure.hpp"
#include "text/to_string.hpp"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        using namespace ncv;
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "randomly warp the input image");
        po_desc.add_options()("input,i",
                boost::program_options::value<ncv::string_t>(),
                "input image path");
        po_desc.add_options()("count,c",
                boost::program_options::value<size_t>()->default_value(32),
                "number of random warpings to generate");
        po_desc.add_options()("translation",
                "use translation fields");
        po_desc.add_options()("rotation",
                "use rotation fields");
        po_desc.add_options()("random",
                "use random fields");
        po_desc.add_options()("alpha",
                boost::program_options::value<ncv::scalar_t>()->default_value(1.0),
                "field mixing maximum coefficient");
        po_desc.add_options()("beta",
                boost::program_options::value<ncv::scalar_t>()->default_value(1.0),
                "gradient magnitue mixing maximum coefficient");
        po_desc.add_options()("save-fields",
                "save fields as image");
        po_desc.add_options()("output,o",
                boost::program_options::value<ncv::string_t>(),
                "output (warped) image path");
	
        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("input") ||
                !po_vm.count("output") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_input = po_vm["input"].as<string_t>();
        const string_t cmd_output = po_vm["output"].as<string_t>();
        const bool cmd_save_fields = po_vm.count("save-fields");

        const auto cmd_count = po_vm["count"].as<size_t>();
        const auto cmd_alpha = po_vm["alpha"].as<scalar_t>();
        const auto cmd_beta = po_vm["beta"].as<scalar_t>();
        const auto cmd_ftype_trs = po_vm.count("translation");
        const auto cmd_ftype_rot = po_vm.count("rotation");
        const auto cmd_ftype_rnd = po_vm.count("random");

        field_type ftype = field_type::random;
        if (cmd_ftype_trs)
        {
                ftype = field_type::translation;
        }
        else if (cmd_ftype_rot)
        {
                ftype = field_type::rotation;
        }
        else if (cmd_ftype_rnd)
        {
                ftype = field_type::random;
        }

        const warp_params params(ftype, 0.1, 4.0, cmd_alpha, cmd_beta);

        // load input image
        image_t iimage;
        ncv::measure_critical_and_log(
               [&] () { return iimage.load_rgba(cmd_input); },
               "loaded image from <" + cmd_input + ">",
               "failed to load image from <" + cmd_input + ">");

        log_info () << "image: " << iimage.cols() << "x" << iimage.rows() << " pixels, "
                    << (iimage.is_luma() ? "[luma]" : "[rgba]") << ".";

        // randomly warp the input image
        for (size_t c = 0; c < cmd_count; c ++)
        {
                // warp
                tensor_t otensor, ftensor;
                ncv::measure_and_log(
                        [&] () { otensor = warp(color::to_rgba_tensor(iimage.rgba()), params, &ftensor); },
                        "warped image");

                // prepare output paths
                const string_t basename =
                        (boost::filesystem::path(cmd_output).parent_path() /
                         boost::filesystem::path(cmd_output).stem()).string();
                const string_t extension =
                         boost::filesystem::path(cmd_output).extension().string();

                const string_t opath = basename + text::to_string(c + 1) + extension;
                const string_t fpath = basename + text::to_string(c + 1) + "_field" + extension;

                // save warped image
                {
                        image_t image;
                        image.load_rgba(color::from_rgba_tensor(otensor));

                        ncv::measure_critical_and_log(
                                [&] () { return image.save(opath); },
                                "saved warped image to <" + opath + ">",
                                "failed to save warped image to <" + opath + ">");
                }

                // save field image
                if (cmd_save_fields)
                {
                        image_t image;
                        image.load_rgba(color::from_rgba_tensor(ftensor));

                        ncv::measure_critical_and_log(
                                [&] () { return image.save(fpath); },
                                "saved field image to <" + fpath + ">",
                                "failed to save field image to <" + fpath + ">");
                }
        }
		
        // OK
        log_info() << ncv::done;
        return EXIT_SUCCESS;
}
