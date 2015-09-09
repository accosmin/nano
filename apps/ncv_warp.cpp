#include "nanocv/string.h"
#include "nanocv/measure.hpp"
#include "nanocv/vision/warp.h"
#include "nanocv/vision/image.h"

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
                boost::program_options::value<ncv::size_t>()->default_value(32),
                "number of random warpings to generate");
        po_desc.add_options()("translation",
                "use translation fields");
        po_desc.add_options()("rotation",
                "use rotation fields");
        po_desc.add_options()("random",
                "use random fields");
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
        const size_t cmd_count = po_vm["count"].as<size_t>();

        const bool cmd_ftype_trs = po_vm.count("translation");
        const bool cmd_ftype_rot = po_vm.count("rotation");
        const bool cmd_ftype_rnd = po_vm.count("random");

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

        const warp_params params(ftype);

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
                tensor_t otensor, ftensor;
                ncv::measure_and_log(
                        [&] () { otensor = warp(color::to_rgba_tensor(iimage.rgba()), params, &ftensor); },
                        "warped image");

                image_t oimage, fimage;
                oimage.load_rgba(color::from_rgba_tensor(otensor));
                fimage.load_rgba(color::from_rgba_tensor(ftensor));

                // save warped image
                const string_t opath =
                        (boost::filesystem::path(cmd_output).parent_path() /
                         boost::filesystem::path(cmd_output).stem()).string() +
                         text::to_string(c + 1) +
                         boost::filesystem::path(cmd_output).extension().string();

                ncv::measure_critical_and_log(
                        [&] () { return oimage.save(opath); },
                        "saved image to <" + opath + ">",
                        "failed to save to <" + opath + ">");

                // save field image
                const string_t fpath =
                        (boost::filesystem::path(cmd_output).parent_path() /
                         boost::filesystem::path(cmd_output).stem()).string() +
                         text::to_string(c + 1) + "_field" +
                         boost::filesystem::path(cmd_output).extension().string();

                ncv::measure_critical_and_log(
                        [&] () { return fimage.save(fpath); },
                        "saved field image to <" + fpath + ">",
                        "failed to save to <" + fpath + ">");
        }
		
        // OK
        log_info() << ncv::done;
        return EXIT_SUCCESS;
}
