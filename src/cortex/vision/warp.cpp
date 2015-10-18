#include "warp.h"
#include "convolve.hpp"
#include "gradient.hpp"
#include "math/gauss.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "tensor/random.hpp"
#include "tensor/transform.hpp"

namespace cortex
{
        namespace
        {
                tensor_t image_field(const matrix_t& fieldx, const matrix_t& fieldy)
                {
                        assert(fieldx.rows() == fieldy.rows());
                        assert(fieldx.cols() == fieldy.cols());

                        const scalar_t pi = std::atan2(0.0, -0.0);
                        const scalar_t ipi = 1.0 / pi;

                        tensor_t image(4, fieldx.rows(), fieldx.cols());

                        tensor::transform(fieldx, fieldy, image.matrix(0), [=] (const scalar_t fx, const scalar_t fy)
                        {
                                return math::clamp(std::sqrt(0.5 * (fx * fx + fy * fy)), 0.0, 1.0);
                        });
                        tensor::transform(fieldx, fieldy, image.matrix(1), [=] (const scalar_t, const scalar_t)
                        {
                                return 0.0;
                        });
                        tensor::transform(fieldx, fieldy, image.matrix(2), [=] (const scalar_t fx, const scalar_t fy)
                        {
                                return 0.5 * math::clamp(0.5 * (ipi * atan2(fy, fx) + 1.0), 0.0, 1.0);
                        });
                        tensor::transform(fieldx, fieldy, image.matrix(3), [] (const scalar_t, const scalar_t)
                        {
                                return 1.0;
                        });

                        return image;
                }

                void smooth_field(matrix_t& field, const scalar_t sigma)
                {
                        const math::gauss_kernel_t<scalar_t> gauss(sigma);
                        cortex::convolve(gauss, field);
                }

                std::tuple<matrix_t, matrix_t> make_random_fields(
                        const size_t rows, const size_t cols,
                        const scalar_t noise, const scalar_t sigma)
                {
                        matrix_t fieldx(rows, cols), fieldy(rows, cols);

                        tensor::set_random(fieldx, math::random_t<scalar_t>(-noise, +noise));
                        tensor::set_random(fieldy, math::random_t<scalar_t>(-noise, +noise));

                        smooth_field(fieldx, sigma);
                        smooth_field(fieldy, sigma);

                        return std::make_tuple(fieldx, fieldy);
                }

                std::tuple<matrix_t, matrix_t> make_translation_fields(
                        const size_t rows, const size_t cols,
                        const scalar_t delta, const scalar_t noise, const scalar_t sigma)
                {
                        matrix_t fieldx(rows, cols), fieldy(rows, cols);

                        tensor::set_random(fieldx, math::random_t<scalar_t>(delta - noise, delta + noise));
                        tensor::set_random(fieldy, math::random_t<scalar_t>(delta - noise, delta + noise));

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

                        math::random_t<scalar_t> rng(-noise, +noise);

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

        warp_params::warp_params(
                field_type ftype,
                scalar_t noise,
                scalar_t sigma,
                scalar_t alpha,
                scalar_t beta)
                :       m_ftype(ftype),
                        m_noise(noise),
                        m_sigma(sigma),
                        m_alpha(alpha),
                        m_beta(beta)
        {
        }

        tensor_t warp(const tensor_t& image, const warp_params& params, tensor_t* fimage)
        {
                tensor_t patch = image;

                // x gradient (directional gradient)
                tensor_t gradx(patch.dims(), patch.rows(), patch.cols());
                for (auto d = 0; d < patch.dims(); d ++)
                {
                        cortex::gradientx(patch.matrix(d), gradx.matrix(d));
                }

                // y gradient (directional gradient)
                tensor_t grady(patch.dims(), patch.rows(), patch.cols());
                for (auto d = 0; d < patch.dims(); d ++)
                {
                        cortex::gradienty(patch.matrix(d), grady.matrix(d));
                }

                // generate random fields
                const scalar_t pi = std::atan2(0.0, -0.0);

                math::random_t<scalar_t> rng_theta(-pi / 8.0, +pi / 8.0);
                math::random_t<scalar_t> rng_delta(-1.0, +1.0);

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
                        make_random_fields(patch.rows(), patch.cols(), 1.0, params.m_sigma);
                        break;
                }

                // visualize the fields (if requested)
                if (fimage)
                {
                        *fimage = image_field(fieldx, fieldy);
                }

                // warp
                math::random_t<scalar_t> rng_alphax(-params.m_alpha, +params.m_alpha);
                math::random_t<scalar_t> rng_alphay(-params.m_alpha, +params.m_alpha);
                math::random_t<scalar_t> rng_beta  (-params.m_beta, +params.m_beta);

                const scalar_t alphax = rng_alphax();
                const scalar_t alphay = rng_alphay();
                const scalar_t beta = rng_beta();

                for (auto d = 0; d < patch.dims(); d ++)
                {
                        warp_by_field(patch.matrix(d),
                                      alphax, fieldx, gradx.matrix(d),
                                      alphay, fieldy, grady.matrix(d),
                                      beta);
                }

                // OK
                return patch;
        }
}
