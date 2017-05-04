#include "warp.h"
#include "gauss.h"
#include "convolve.h"
#include "gradient.h"
#include "math/random.h"
#include "math/numeric.h"
#include "tensor/numeric.h"

namespace nano
{
        static constexpr auto one = scalar_t(1);
        static constexpr auto half = scalar_t(0.5);
        static const auto pi = 4 * std::atan(one);

        template <typename trng>
        static void make_random_fields(trng& rng, const scalar_t noise,
                matrix_t& fieldx, matrix_t& fieldy)
        {
                nano::set_random([&] () { return rng() * noise; }, fieldx, fieldy);
        }

        template <typename trng>
        static void make_translation_fields(trng& rng, const scalar_t delta, const scalar_t noise,
                matrix_t& fieldx, matrix_t& fieldy)
        {
                nano::set_random([&] () { return rng() * noise + delta; }, fieldx, fieldy);
        }

        template <typename trng>
        static void make_rotation_fields(trng& rng, const scalar_t delta, const scalar_t theta, const scalar_t noise,
                matrix_t& fieldx, matrix_t& fieldy)
        {
                const auto rows = fieldx.rows(), cols = fieldx.cols();
                const auto cx = half * static_cast<scalar_t>(cols);
                const auto cy = half * static_cast<scalar_t>(rows);
                const auto id = one / (nano::square(cx) + nano::square(cy));
                const auto cos_theta = std::cos(theta);
                const auto sin_theta = std::sin(theta);

                for (tensor_size_t r = 0; r < rows; ++ r)
                {
                        for (tensor_size_t c = 0; c < cols; ++ c)
                        {
                                const auto dist = nano::square(scalar_t(r) - cy) + nano::square(scalar_t(c) - cx);

                                fieldx(r, c) = id * dist * cos_theta + rng() * noise + delta;
                                fieldy(r, c) = id * dist * sin_theta + rng() * noise + delta;
                        }
                }
        }

        template <typename tmatrixio, typename tmatrixg>
        static void warp_by_field(tmatrixio&& iodata,
                const scalar_t alphax, const matrix_t& fieldx, const tmatrixg& gradx,
                const scalar_t alphay, const matrix_t& fieldy, const tmatrixg& grady,
                const scalar_t beta)
        {
                iodata.array() +=
                        alphax * fieldx.array() * gradx.array() +
                        alphay * fieldy.array() * grady.array() +
                        beta * (gradx.array().square() + grady.array().square()).sqrt();
        }

        void warp(tensor3d_t& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta)
        {
                const auto imaps = iodata.size<0>();
                const auto irows = iodata.size<1>();
                const auto icols = iodata.size<2>();

                auto rng = make_rng<scalar_t>(-1, +1);

                // generate random fields
                const auto delta = rng() * one;
                const auto theta = rng() * pi / 8;

                matrix_t fieldx(irows, icols);
                matrix_t fieldy(irows, icols);
                switch (wtype)
                {
                case warp_type::translation:
                        make_translation_fields(rng, delta, noise, fieldx, fieldy);
                        break;

                case warp_type::rotation:
                        make_rotation_fields(rng, 0, theta, noise, fieldx, fieldy);
                        break;

                case warp_type::mixed:
                        make_rotation_fields(rng, delta, theta, noise, fieldx, fieldy);
                        break;

                case warp_type::random:
                default:
                        make_random_fields(rng, noise, fieldx, fieldy);
                        break;
                }

                // smooth fields
                const auto gauss = make_gauss_kernel(sigma);
                nano::convolve(gauss, fieldx);
                nano::convolve(gauss, fieldy);

                // mix input image with field-weighted gradients
                const auto alphax = rng() * alpha;
                const auto alphay = rng() * alpha;
                const auto betamx = rng() * beta;

                for (auto d = 0; d < imaps; ++ d)
                {
                        const auto gradx = nano::gradientx(iodata.matrix(d));
                        const auto grady = nano::gradienty(iodata.matrix(d));

                        warp_by_field(iodata.matrix(d), alphax, fieldx, gradx, alphay, fieldy, grady, betamx);
                }
        }
}
