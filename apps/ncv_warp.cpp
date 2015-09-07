#include "nanocv/string.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/gauss.hpp"
#include "nanocv/vision/image.h"
#include "nanocv/math/random.hpp"
#include "nanocv/tensor/random.hpp"
#include "nanocv/vision/gradient.hpp"
#include "nanocv/vision/convolve.hpp"
#include <boost/program_options.hpp>

namespace
{
        using namespace ncv;

        /// \todo move these to library (once the warping algorithm works OK)

        matrix_t make_random_field(
                const size_t rows, const size_t cols,
                const scalar_t min_delta, const scalar_t max_delta, const scalar_t sigma)
        {
                matrix_t field(rows, cols);

                random_t<scalar_t> rng(min_delta, max_delta);
                tensor::set_random(field, rng);

                const gauss_kernel_t<scalar_t> gauss(sigma);
                ncv::convolve(gauss, field);

                return field;
        }

        matrix_t make_translation_field(
                const size_t rows, const size_t cols,
                const scalar_t delta, const scalar_t max_noise, const scalar_t sigma)
        {
                return make_random_field(rows, cols, delta - max_noise, delta + max_noise, sigma);
        }

        template
        <
                typename tmatrixio,
                typename tmatrixf,
                typename tmatrixt
        >
        void warp_by_field(tmatrixio&& x,
                const scalar_t alphax, const tmatrixf& fieldx, const tmatrixt& gradx,
                const scalar_t alphay, const tmatrixf& fieldy, const tmatrixt& grady,
                const scalar_t beta)
        {
                x.array() +=
                        alphax * fieldx.array() * gradx.array() +
                        alphay * fieldy.array() * grady.array() +
                        beta * (fieldx.array().square() + fieldy.array().square()).sqrt();
        }

        image_t warp(const image_t& iimage)
        {
                tensor_t patch = ncv::color::to_rgba_tensor(iimage.rgba());

                // x gradient (directional gradient)
                tensor_t patchgx(4, patch.rows(), patch.cols());
                ncv::gradientx(patch.matrix(0), patchgx.matrix(0));
                ncv::gradientx(patch.matrix(1), patchgx.matrix(1));
                ncv::gradientx(patch.matrix(2), patchgx.matrix(2));
                ncv::gradientx(patch.matrix(3), patchgx.matrix(3));

                // y gradient (directional gradient)
                tensor_t patchgy(4, patch.rows(), patch.cols());
                ncv::gradienty(patch.matrix(0), patchgy.matrix(0));
                ncv::gradienty(patch.matrix(1), patchgy.matrix(1));
                ncv::gradienty(patch.matrix(2), patchgy.matrix(2));
                ncv::gradienty(patch.matrix(3), patchgy.matrix(3));

                // generate random fields
                random_t<scalar_t> rng_delta(-4.0, +4.0);
                random_t<scalar_t> rng_noise(+0.0, +4.0);
                random_t<scalar_t> rng_sigma(+0.0, +2.0);

                const matrix_t fieldx = make_translation_field(patch.rows(), patch.cols(), rng_delta(), rng_noise(), rng_sigma());
                const matrix_t fieldy = make_translation_field(patch.rows(), patch.cols(), rng_delta(), rng_noise(), rng_sigma());

                // warp
                random_t<scalar_t> rng_alpha(-1.0, +1.0);
                random_t<scalar_t> rng_beta(-1.0, +1.0);

                const scalar_t alphax = rng_alpha();
                const scalar_t alphay = rng_alpha();
                const scalar_t beta = rng_beta();

                warp_by_field(patch.matrix(0), alphax, fieldx, patchgx.matrix(0), alphay, fieldy, patchgy.matrix(0), beta);
                warp_by_field(patch.matrix(1), alphax, fieldx, patchgx.matrix(1), alphay, fieldy, patchgy.matrix(1), beta);
                warp_by_field(patch.matrix(2), alphax, fieldx, patchgx.matrix(2), alphay, fieldy, patchgy.matrix(2), beta);
                warp_by_field(patch.matrix(3), alphax, fieldx, patchgx.matrix(3), alphay, fieldy, patchgy.matrix(3), beta);

                // OK
                image_t oimage;
                oimage.load_rgba(color::from_rgba_tensor(patch));
                return oimage;
        }
}

int main(int argc, char *argv[])
{
        using namespace ncv;
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "randomly warp the input image");
        po_desc.add_options()("input,i",
                boost::program_options::value<ncv::string_t>(),
                "input image path");
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

        // load input image
        image_t iimage;
        ncv::measure_critical_and_log(
               [&] () { return iimage.load_rgba(cmd_input); },
               "loaded image from <" + cmd_input + ">",
               "failed to load image from <" + cmd_input + ">");

        log_info () << "image: " << iimage.cols() << "x" << iimage.rows() << " pixels, "
                    << (iimage.is_luma() ? "[luma]" : "[rgba]") << ".";

        // randomly warp the input image
        image_t oimage;
        ncv::measure_and_log(
                [&] () { oimage = warp(iimage); },
                "warped image");

        // save output image
        ncv::measure_critical_and_log(
                [&] () { return oimage.save(cmd_output); },
                "saved image to <" + cmd_output + ">",
                "failed to save to <" + cmd_output + ">");
		
        // OK
        log_info() << ncv::done;
        return EXIT_SUCCESS;
}
