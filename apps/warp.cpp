#include "stringi.h"
#include "text/cmdline.h"
#include "vision/warp.h"
#include "vision/image.h"
#include "text/filesystem.h"
#include "text/to_string.hpp"
#include "cortex/util/measure_and_log.hpp"

int main(int argc, char *argv[])
{
        using namespace cortex;

        // parse the command line
        text::cmdline_t cmdline("randomly warp the input image");
        cmdline.add("i", "input",       "input image path");
        cmdline.add("c", "count",       "number of random warpings to generate", "32");
        cmdline.add("", "translation",  "use translation fields");
        cmdline.add("", "rotation",     "use rotation fields");
        cmdline.add("", "random",       "use random fields");
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
        const auto cmd_ftype_trs = cmdline.has("translation");
        const auto cmd_ftype_rot = cmdline.has("rotation");
        const auto cmd_ftype_rnd = cmdline.has("random");

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
        cortex::measure_critical_and_log(
               [&] () { return iimage.load_rgba(cmd_input); },
               "load image from <" + cmd_input + ">");

        log_info () << "image: " << iimage.cols() << "x" << iimage.rows() << " pixels, "
                    << (iimage.is_luma() ? "[luma]" : "[rgba]") << ".";

        // randomly warp the input image
        for (size_t c = 0; c < cmd_count; ++ c)
        {
                // warp
                tensor_t otensor, ftensor;
                cortex::measure_and_log(
                        [&] () { otensor = warp(color::to_rgba_tensor(iimage.rgba()), params, &ftensor); },
                        "warped image");

                // prepare output paths
                const string_t basename = text::dirname(cmd_output) + text::stem(cmd_output);
                const string_t extension = text::extension(cmd_output);

                const string_t opath = basename + text::to_string(c + 1) + extension;
                const string_t fpath = basename + text::to_string(c + 1) + "_field" + extension;

                // save warped image
                {
                        image_t image;
                        image.load_rgba(color::from_rgba_tensor(otensor));

                        cortex::measure_critical_and_log(
                                [&] () { return image.save(opath); },
                                "save warped image to <" + opath + ">");
                }

                // save field image
                if (cmd_save_fields)
                {
                        image_t image;
                        image.load_rgba(color::from_rgba_tensor(ftensor));

                        cortex::measure_critical_and_log(
                                [&] () { return image.save(fpath); },
                                "save field image to <" + fpath + ">");
                }
        }

        // OK
        log_info() << cortex::done;
        return EXIT_SUCCESS;
}
