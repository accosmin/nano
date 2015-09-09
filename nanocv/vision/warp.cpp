#include "warp.h"
#include "convolve.hpp"
#include "gradient.hpp"
#include "nanocv/math/gauss.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/math/numeric.hpp"
#include "nanocv/tensor/random.hpp"
#include "nanocv/tensor/transform.hpp"

namespace ncv
{
        namespace
        {
                image_t image_field(const matrix_t& fieldx, const matrix_t& fieldy)
                {
                        assert(fieldx.rows() == fieldy.rows());
                        assert(fieldx.cols() == fieldy.cols());

                        rgba_matrix_t rgba(fieldx.rows(), fieldx.cols());
                        tensor::transform(fieldx, fieldy, rgba, [] (const scalar_t fx, const scalar_t fy)
                        {
                                const auto red  = math::clamp(255.0 * 0.5 * (fx + 1.0), 0.0, 255.0);
                                const auto blue = math::clamp(255.0 * 0.5 * (fy + 1.0), 0.0, 255.0);

                                return color::make_rgba(
                                        math::cast<rgba_t>(red),
                                        0,
                                        math::cast<rgba_t>(blue));
                        });

                        image_t image;
                        image.load_rgba(rgba);
                        return image;
                }

                void smooth_field(matrix_t& field, const scalar_t sigma)
                {
                        const gauss_kernel_t<scalar_t> gauss(sigma);
                        ncv::convolve(gauss, field);
                }

                std::tuple<matrix_t, matrix_t> make_random_fields(
                        const size_t rows, const size_t cols,
                        const scalar_t noise, const scalar_t sigma)
                {
                        matrix_t fieldx(rows, cols), fieldy(rows, cols);

                        tensor::set_random(fieldx, random_t<scalar_t>(-noise, +noise));
                        tensor::set_random(fieldy, random_t<scalar_t>(-noise, +noise));

                        smooth_field(fieldx, sigma);
                        smooth_field(fieldy, sigma);

                        return std::make_tuple(fieldx, fieldy);
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
                        const scalar_t theta, const scalar_t noise, const scalar_t sigma)
                {
                        matrix_t fieldx(rows, cols), fieldy(rows, cols);

                        const scalar_t cx = 0.5 * cols;
                        const scalar_t cy = 0.5 * rows;
                        const scalar_t id = 1.0 / (math::square(cx) + math::square(cy));

                        random_t<scalar_t> rng(-noise, +noise);

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
        }

        image_t warp(const image_t& image, const warp_params& params, image_t* fimage)
        {
                assert(image.is_rgba());
                tensor_t patch = ncv::color::to_rgba_tensor(image.rgba());

                // x gradient (directional gradient)
                tensor_t gradx(4, patch.rows(), patch.cols());
                ncv::gradientx(patch.matrix(0), gradx.matrix(0));
                ncv::gradientx(patch.matrix(1), gradx.matrix(1));
                ncv::gradientx(patch.matrix(2), gradx.matrix(2));
                ncv::gradientx(patch.matrix(3), gradx.matrix(3));

                // y gradient (directional gradient)
                tensor_t grady(4, patch.rows(), patch.cols());
                ncv::gradienty(patch.matrix(0), grady.matrix(0));
                ncv::gradienty(patch.matrix(1), grady.matrix(1));
                ncv::gradienty(patch.matrix(2), grady.matrix(2));
                ncv::gradienty(patch.matrix(3), grady.matrix(3));

                // generate random fields
                const scalar_t pi = std::atan2(0.0, -0.0);

                random_t<scalar_t> rng_theta(-pi / 8.0, +pi / 8.0);
                random_t<scalar_t> rng_delta(-1.0, +1.0);

                matrix_t fieldx, fieldy;
                switch (params.m_ftype)
                {
                case field_type::translation:
                        std::tie(fieldx, fieldy) =
                        make_translation_fields(patch.rows(), patch.cols(), rng_delta(), params.m_noise, params.m_sigma);
                        break;

                case field_type::rotation:
                        std::tie(fieldx, fieldy) =
                        make_rotation_fields(patch.rows(), patch.cols(), rng_theta(), params.m_noise, params.m_sigma);
                        break;

                case field_type::random:
                default:
                        std::tie(fieldx, fieldy) =
                        make_random_fields(patch.rows(), patch.cols(), params.m_noise, params.m_sigma);
                        break;
                }

                // visualize the fields (if requested)
                if (fimage)
                {
                        *fimage = image_field(fieldx, fieldy);
                }

                // warp
                random_t<scalar_t> rng_alphax(-1.0, +1.0);
                random_t<scalar_t> rng_alphay(-1.0, +1.0);
                random_t<scalar_t> rng_beta  (-1.0, +1.0);

                const scalar_t alphax = rng_alphax();
                const scalar_t alphay = rng_alphay();
                const scalar_t beta = rng_beta();

                warp_by_field(patch.matrix(0), alphax, fieldx, gradx.matrix(0), alphay, fieldy, grady.matrix(0), beta);
                warp_by_field(patch.matrix(1), alphax, fieldx, gradx.matrix(1), alphay, fieldy, grady.matrix(1), beta);
                warp_by_field(patch.matrix(2), alphax, fieldx, gradx.matrix(2), alphay, fieldy, grady.matrix(2), beta);
                warp_by_field(patch.matrix(3), alphax, fieldx, gradx.matrix(3), alphay, fieldy, grady.matrix(3), beta);

                // OK
                image_t oimage;
                oimage.load_rgba(color::from_rgba_tensor(patch));
                return oimage;
        }
}
