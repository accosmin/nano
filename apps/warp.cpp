#include "stringi.h"
#include "text/cmdline.h"
#include "vision/image.h"
#include "text/to_params.h"
#include "measure_and_log.h"
#include "text/filesystem.h"
#include "text/concatenate.h"
#include "samplers/sampler_warp.h"

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("randomly warp the input image");
        cmdline.add("i", "input",       "input image path");
        cmdline.add("c", "count",       "number of random warpings to generate", "32");
        cmdline.add("", "type",         "warping type [" + concatenate(enum_values<warp_type>()) + "]");
        cmdline.add("", "alpha",        "field mixing coefficient", "1.0");
        cmdline.add("", "beta",         "gradient magnitue mixing coefficient", "1.0");
        cmdline.add("", "save-fields",  "save fields as image");
        cmdline.add("o", "output",      "output (warped) image path");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_input = cmdline.get<string_t>("input");
        const auto cmd_output = cmdline.get<string_t>("output");
        const auto cmd_save_fields = cmdline.has("save-fields");
        const auto cmd_count = cmdline.get<size_t>("count");
        const auto cmd_alpha = cmdline.get<scalar_t>("alpha");
        const auto cmd_beta = cmdline.get<scalar_t>("beta");
        const auto cmd_type = cmdline.get<warp_type>("type");

        auto warper = sampler_warp_t(to_params(
                "type", cmd_type, "noise", 0.1, "sigma", 4.0, "alpha", cmd_alpha, "beta", cmd_beta));

        // load input image
        image_t iimage;
        nano::measure_critical_and_log(
               [&] () { return iimage.load_rgba(cmd_input); },
               "load image from <" + cmd_input + ">");

        log_info () << "image: " << iimage.cols() << "x" << iimage.rows() << " pixels, "
                    << (iimage.is_luma() ? "[luma]" : "[rgba]") << ".";

        // randomly warp the input image
        for (size_t c = 0; c < cmd_count; ++ c)
        {
                // warp
                tensor3d_t wtensor = iimage.to_tensor();
                nano::measure_and_log(
                        [&] () { warper.get(wtensor, nullptr, nullptr); },
                        "warped image");

                // prepare output paths
                const string_t basename = nano::dirname(cmd_output) + nano::stem(cmd_output);
                const string_t extension = nano::extension(cmd_output);

                const string_t opath = basename + nano::to_string(c + 1) + "." + extension;
                const string_t fpath = basename + nano::to_string(c + 1) + "_field." + extension;

                // save warped image
                {
                        image_t image;
                        image.from_tensor(wtensor);
                        image.make_rgba();

                        nano::measure_critical_and_log(
                                [&] () { return image.save(opath); },
                                "save warped image to <" + opath + ">");
                }

                // save field image
                if (cmd_save_fields)
                {
                        image_t image;
                        image.from_tensor(warper.fimage());
                        image.make_rgba();

                        nano::measure_critical_and_log(
                                [&] () { return image.save(fpath); },
                                "save field image to <" + fpath + ">");
                }
        }

        // OK
        log_info() << nano::done;
        return EXIT_SUCCESS;
}
