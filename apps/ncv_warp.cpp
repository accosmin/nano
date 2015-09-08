#include "nanocv/string.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/gauss.hpp"
#include "nanocv/vision/image.h"
#include "nanocv/math/random.hpp"
#include "nanocv/math/numeric.hpp"
#include "nanocv/tensor/random.hpp"
#include "nanocv/vision/gradient.hpp"
#include "nanocv/vision/convolve.hpp"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace
{
        using namespace ncv;

        /// \todo move these to library (once the warping algorithm works OK)

        void smooth_field(matrix_t& field, const scalar_t sigma)
        {
                const gauss_kernel_t<scalar_t> gauss(sigma);
                ncv::convolve(gauss, field);
        }

        matrix_t make_random_field(
                const size_t rows, const size_t cols, random_t<scalar_t> rng, const scalar_t sigma)
        {
                matrix_t field(rows, cols);

                tensor::set_random(field, rng);

                smooth_field(field, sigma);

                return field;
        }

        std::tuple<matrix_t, matrix_t> make_translation_fields(
                const size_t rows, const size_t cols,
                const scalar_t delta, const scalar_t noise, const scalar_t sigma)
        {
                matrix_t fieldx(rows, cols), fieldy(rows, cols);

                tensor::set_random(fieldx, random_t<scalar_t>(delta - noise, delta + noise));
                tensor::set_random(fieldy, random_t<scalar_t>(delta - noise, delta + noise));

                smooth_field(fieldx, sigma);
                smooth_field(fieldy, sigma);

                return std::make_tuple(fieldx, fieldy);
        }

        std::tuple<matrix_t, matrix_t> make_rotation_fields(
                const size_t rows, const size_t cols,
                const scalar_t delta,
                const scalar_t theta,
                const scalar_t sigma)
        {
                matrix_t fieldx(rows, cols), fieldy(rows, cols);

                const scalar_t cx = 0.5 * cols;
                const scalar_t cy = 0.5 * rows;
                const scalar_t id = 1.0 / (math::square(cx) + math::square(cy));

                random_t<scalar_t> rng(-delta, +delta);

                for (size_t r = 0; r < rows; r ++)
                {
                        for (size_t c = 0; c < cols; c ++)
                        {
                                const auto dist = math::square(scalar_t(r) - cy) + math::square(scalar_t(c) - cx);

                                fieldx(r, c) = id * dist * std::cos(theta) + rng();
                                fieldy(r, c) = id * dist * std::sin(theta) + rng();
                        }
                }

                smooth_field(fieldx, sigma);
                smooth_field(fieldy, sigma);

                return std::make_tuple(fieldx, fieldy);
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
                        beta * (gradx.array().square() + grady.array().square()).sqrt();
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
                const scalar_t pi = std::atan2(0.0, -0.0);
                const auto max_delta = scalar_t(std::min(iimage.rows(), iimage.cols())) / 16.0;
                const auto max_sigma = scalar_t(std::min(iimage.rows(), iimage.cols())) / 16.0;

                random_t<scalar_t> rng_theta(-pi / 8.0, +pi / 8.0);
                random_t<scalar_t> rng_delta(-max_delta, +max_delta);
                random_t<scalar_t> rng_sigma(+1.0, +1.0 + max_sigma);
                random_t<scalar_t> rng_noise(+0.0, max_delta / 4.0);

                matrix_t fieldx, fieldy;

//                std::tie(fieldx, fieldy) = make_translation_fields(patch.rows(), patch.cols(),
//                        rng_delta(), rng_noise(), rng_sigma());

                std::tie(fieldx, fieldy) = make_rotation_fields(patch.rows(), patch.cols(),
                        /*rng_delta()*/ 0.0, rng_theta(), rng_sigma());

                // warp
                random_t<scalar_t> rng_alpha(-1.0, +1.0);
                random_t<scalar_t> rng_beta (-1.0, +1.0);

                const scalar_t alphax = 8.0 * rng_alpha();
                const scalar_t alphay = 8.0 * rng_alpha();
                const scalar_t beta = 0.0;//rng_beta();

                log_info() << "patch(0) = [" << patch.matrix(0).minCoeff() << ", " << patch.matrix(0).maxCoeff() << "]";
                log_info() << "patch(1) = [" << patch.matrix(1).minCoeff() << ", " << patch.matrix(1).maxCoeff() << "]";
                log_info() << "patch(2) = [" << patch.matrix(2).minCoeff() << ", " << patch.matrix(2).maxCoeff() << "]";
                log_info() << "patch(3) = [" << patch.matrix(3).minCoeff() << ", " << patch.matrix(3).maxCoeff() << "]";

                warp_by_field(patch.matrix(0), alphax, fieldx, patchgx.matrix(0), alphay, fieldy, patchgy.matrix(0), beta);
                warp_by_field(patch.matrix(1), alphax, fieldx, patchgx.matrix(1), alphay, fieldy, patchgy.matrix(1), beta);
                warp_by_field(patch.matrix(2), alphax, fieldx, patchgx.matrix(2), alphay, fieldy, patchgy.matrix(2), beta);
                warp_by_field(patch.matrix(3), alphax, fieldx, patchgx.matrix(3), alphay, fieldy, patchgy.matrix(3), beta);

                log_info() << "*patch(0) = [" << patch.matrix(0).minCoeff() << ", " << patch.matrix(0).maxCoeff() << "]";
                log_info() << "*patch(1) = [" << patch.matrix(1).minCoeff() << ", " << patch.matrix(1).maxCoeff() << "]";
                log_info() << "*patch(2) = [" << patch.matrix(2).minCoeff() << ", " << patch.matrix(2).maxCoeff() << "]";
                log_info() << "*patch(3) = [" << patch.matrix(3).minCoeff() << ", " << patch.matrix(3).maxCoeff() << "]";

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
        po_desc.add_options()("count,c",
                boost::program_options::value<ncv::size_t>()->default_value(32),
                "number of random warpings to generate");
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
                const string_t path =
                        (boost::filesystem::path(cmd_output).parent_path() /
                         boost::filesystem::path(cmd_output).stem()).string() +
                         text::to_string(c + 1) +
                         boost::filesystem::path(cmd_output).extension().string();

                image_t oimage;
                ncv::measure_and_log(
                        [&] () { oimage = warp(iimage); },
                        "warped image");

                ncv::measure_critical_and_log(
                        [&] () { return oimage.save(path); },
                        "saved image to <" + path + ">",
                        "failed to save to <" + path + ">");
        }
		
        // OK
        log_info() << ncv::done;
        return EXIT_SUCCESS;
}