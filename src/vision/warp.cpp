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
        static constexpr auto one = scalar_t(1);
        static constexpr auto zero = scalar_t(0);
        static constexpr auto half = scalar_t(0.5);
        static const auto pi = 4 * std::atan(one);
        static const auto ipi = one / pi;

        static void image_field(const matrix_t& fieldx, const matrix_t& fieldy, tensor3d_t& image)
        {
                assert(fieldx.rows() == fieldy.rows());
                assert(fieldx.cols() == fieldy.cols());

                image.resize(4, fieldx.rows(), fieldx.cols());
                nano::transform(fieldx, fieldy, image.matrix(0), [=] (const scalar_t fx, const scalar_t fy)
                {
                        return nano::clamp(std::sqrt(half * (fx * fx + fy * fy)), zero, one);
                });
                nano::transform(fieldx, fieldy, image.matrix(1), [=] (const scalar_t, const scalar_t)
                {
                        return zero;
                });
                nano::transform(fieldx, fieldy, image.matrix(2), [=] (const scalar_t fx, const scalar_t fy)
                {
                        return half * nano::clamp(half * (ipi * atan2(fy, fx) + one), zero, one);
                });
                nano::transform(fieldx, fieldy, image.matrix(3), [] (const scalar_t, const scalar_t)
                {
                        return one;
                });
        }

        static void smooth_field(matrix_t& field, const scalar_t sigma)
        {
                const nano::gauss_kernel_t<scalar_t> gauss(sigma);
                nano::convolve(gauss, field);
        }

        static void make_random_fields(const scalar_t noise, const scalar_t sigma,
                matrix_t& fieldx, matrix_t& fieldy)
        {
                nano::set_random(nano::make_rng<scalar_t>(-noise, +noise), fieldx, fieldy);

                smooth_field(fieldx, sigma);
                smooth_field(fieldy, sigma);
        }

        static void make_translation_fields(const scalar_t delta, const scalar_t noise, const scalar_t sigma,
                matrix_t& fieldx, matrix_t& fieldy)
        {
                nano::set_random(nano::make_rng<scalar_t>(delta - noise, delta + noise), fieldx, fieldy);

                smooth_field(fieldx, sigma);
                smooth_field(fieldy, sigma);
        }

        static void make_rotation_fields(const scalar_t theta, const scalar_t noise, const scalar_t sigma,
                matrix_t& fieldx, matrix_t& fieldy)
        {
                const auto rows = fieldx.rows(), cols = fieldx.cols();
                const auto cx = half * static_cast<scalar_t>(cols);
                const auto cy = half * static_cast<scalar_t>(rows);
                const auto id = one / (nano::square(cx) + nano::square(cy));

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
        }

        template <typename tmatrixi, typename tmatrixg, typename tmatrixo>
        static void warp_by_field(const tmatrixi& idata,
                const scalar_t alphax, const matrix_t& fieldx, const tmatrixg& gradx,
                const scalar_t alphay, const matrix_t& fieldy, const tmatrixg& grady,
                const scalar_t beta,
                tmatrixo&& odata)
        {
                odata.array() = idata.array() +
                        alphax * fieldx.array() * gradx.array() +
                        alphay * fieldy.array() * grady.array() +
                        beta * (gradx.array().square() + grady.array().square()).sqrt();
        }

        warper_t::warper_t(
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

        void warper_t::operator()(const tensor3d_t& idata, tensor3d_t& odata, tensor3d_t* fimage)
        {
                const auto imaps = idata.size<0>();
                const auto irows = idata.size<1>();
                const auto icols = idata.size<2>();

                // x gradient (directional gradient)
                m_gradx.resize(imaps, irows, icols);
                for (auto d = 0; d < imaps; ++ d)
                {
                        nano::gradientx(idata.matrix(d), m_gradx.matrix(d));
                }

                // y gradient (directional gradient)
                m_grady.resize(imaps, irows, icols);
                for (auto d = 0; d < imaps; ++ d)
                {
                        nano::gradienty(idata.matrix(d), m_grady.matrix(d));
                }

                // generate random fields
                auto rng_theta = nano::make_rng<scalar_t>(-pi / 8, +pi / 8);
                auto rng_delta = nano::make_rng<scalar_t>(-one, +one);

                m_fieldx.resize(irows, icols);
                m_fieldy.resize(irows, icols);
                switch (m_ftype)
                {
                case field_type::translation:
                        make_translation_fields(rng_delta(), m_noise, m_sigma, m_fieldx, m_fieldy);
                        break;

                case field_type::rotation:
                        make_rotation_fields(rng_theta(), m_noise, m_sigma, m_fieldx, m_fieldy);
                        break;

                case field_type::random:
                default:
                        make_random_fields(m_noise, m_sigma, m_fieldx, m_fieldy);
                        break;
                }

                // visualize the fields (if requested)
                if (fimage)
                {
                        image_field(m_fieldx, m_fieldy, *fimage);
                }

                // warp
                auto rng_alphax = nano::make_rng<scalar_t>(-m_alpha, +m_alpha);
                auto rng_alphay = nano::make_rng<scalar_t>(-m_alpha, +m_alpha);
                auto rng_beta   = nano::make_rng<scalar_t>(-m_beta, +m_beta);

                const auto alphax = rng_alphax();
                const auto alphay = rng_alphay();
                const auto beta = rng_beta();

                odata.resize(imaps, irows, icols);
                for (auto d = 0; d < imaps; ++ d)
                {
                        warp_by_field(idata.matrix(d),
                                      alphax, m_fieldx, m_gradx.matrix(d),
                                      alphay, m_fieldy, m_grady.matrix(d),
                                      beta,
                                      odata.matrix(d));
                }
        }
}
