#include "warp.h"
#include "convolve.h"
#include "gradient.h"
#include "math/gauss.h"
#include "math/random.h"
#include "math/numeric.h"
#include "tensor/numeric.h"
#include "tensor/algorithm.h"

namespace nano
{
        static const scalar_t one = scalar_t(1);
        static const scalar_t zero = scalar_t(0);
        static const scalar_t half = scalar_t(0.5);

        static tensor3d_t image_field(const matrix_t& fieldx, const matrix_t& fieldy)
        {
                assert(fieldx.rows() == fieldy.rows());
                assert(fieldx.cols() == fieldy.cols());

                const scalar_t pi = 4 * std::atan(one);
                const scalar_t ipi = one / pi;

                tensor3d_t image(4, fieldx.rows(), fieldx.cols());

                tensor::transform(fieldx, fieldy, image.matrix(0), [=] (const scalar_t fx, const scalar_t fy)
                {
                        return nano::clamp(std::sqrt(half * (fx * fx + fy * fy)), zero, one);
                });
                tensor::transform(fieldx, fieldy, image.matrix(1), [=] (const scalar_t, const scalar_t)
                {
                        return zero;
                });
                tensor::transform(fieldx, fieldy, image.matrix(2), [=] (const scalar_t fx, const scalar_t fy)
                {
                        return half * nano::clamp(half * (ipi * atan2(fy, fx) + one), zero, one);
                });
                tensor::transform(fieldx, fieldy, image.matrix(3), [] (const scalar_t, const scalar_t)
                {
                        return one;
                });

                return image;
        }

        static void smooth_field(matrix_t& field, const scalar_t sigma)
        {
                const nano::gauss_kernel_t<scalar_t> gauss(sigma);
                nano::convolve(gauss, field);
        }

        static std::tuple<matrix_t, matrix_t> make_random_fields(
                const tensor_size_t rows, const tensor_size_t cols,
                const scalar_t noise, const scalar_t sigma)
        {
                matrix_t fieldx(rows, cols), fieldy(rows, cols);

                tensor::set_random(nano::make_rng<scalar_t>(-noise, +noise), fieldx, fieldy);

                smooth_field(fieldx, sigma);
                smooth_field(fieldy, sigma);

                return std::make_tuple(fieldx, fieldy);
        }

        static std::tuple<matrix_t, matrix_t> make_translation_fields(
                const tensor_size_t rows, const tensor_size_t cols,
                const scalar_t delta, const scalar_t noise, const scalar_t sigma)
        {
                matrix_t fieldx(rows, cols), fieldy(rows, cols);

                tensor::set_random(nano::make_rng<scalar_t>(delta - noise, delta + noise), fieldx, fieldy);

                smooth_field(fieldx, sigma);
                smooth_field(fieldy, sigma);

                return std::make_tuple(fieldx, fieldy);
        }

        static std::tuple<matrix_t, matrix_t> make_rotation_fields(
                const tensor_size_t rows, const tensor_size_t cols,
                const scalar_t theta, const scalar_t noise, const scalar_t sigma)
        {
                matrix_t fieldx(rows, cols), fieldy(rows, cols);

                const scalar_t cx = half * static_cast<scalar_t>(cols);
                const scalar_t cy = half * static_cast<scalar_t>(rows);
                const scalar_t id = one / (nano::square(cx) + nano::square(cy));

                auto rng = nano::make_rng<scalar_t>(-noise, +noise);

                for (tensor_size_t r = 0; r < rows; ++ r)
                {
                        for (tensor_size_t c = 0; c < cols; ++ c)
                        {
                                const auto dist = nano::square(scalar_t(r) - cy) + nano::square(scalar_t(c) - cx);

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
        static void warp_by_field(tmatrixio&& x,
                const scalar_t alphax, const tmatrixf& fieldx, const tmatrixt& gradx,
                const scalar_t alphay, const tmatrixf& fieldy, const tmatrixt& grady,
                const scalar_t beta)
        {
                x.array() +=
                        alphax * fieldx.array() * gradx.array() +
                        alphay * fieldy.array() * grady.array() +
                        beta * (gradx.array().square() + grady.array().square()).sqrt();
        }

        warp_params_t::warp_params_t(
                field_type ftype,
                scalar_t noise,
                scalar_t sigma,
                scalar_t alpha,
                scalar_t beta) :
                m_ftype(ftype),
                m_noise(noise),
                m_sigma(sigma),
                m_alpha(alpha),
                m_beta(beta)
        {
        }

        tensor3d_t warp(const tensor3d_t& image, const warp_params_t& params, tensor3d_t* fimage)
        {
                assert(image.size<0>() == 4);

                tensor3d_t patch = image;

                // x gradient (directional gradient)
                tensor3d_t gradx(patch.size<0>(), patch.rows(), patch.cols());
                for (auto d = 0; d < patch.size<0>(); ++ d)
                {
                        nano::gradientx(patch.matrix(d), gradx.matrix(d));
                }

                // y gradient (directional gradient)
                tensor3d_t grady(patch.size<0>(), patch.rows(), patch.cols());
                for (auto d = 0; d < patch.size<0>(); ++ d)
                {
                        nano::gradienty(patch.matrix(d), grady.matrix(d));
                }

                // generate random fields
                const scalar_t pi = 4 * std::atan(one);

                auto rng_theta = nano::make_rng<scalar_t>(-pi / 8, +pi / 8);
                auto rng_delta = nano::make_rng<scalar_t>(-one, +one);

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
                        make_random_fields(patch.rows(), patch.cols(), one, params.m_sigma);
                        break;
                }

                // visualize the fields (if requested)
                if (fimage)
                {
                        *fimage = image_field(fieldx, fieldy);
                }

                // warp
                nano::random_t<scalar_t> rng_alphax(-params.m_alpha, +params.m_alpha);
                nano::random_t<scalar_t> rng_alphay(-params.m_alpha, +params.m_alpha);
                nano::random_t<scalar_t> rng_beta  (-params.m_beta, +params.m_beta);

                const scalar_t alphax = rng_alphax();
                const scalar_t alphay = rng_alphay();
                const scalar_t beta = rng_beta();

                for (auto d = 0; d < patch.size<0>(); ++ d)
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
