#include "warp.h"
#include "gauss.h"
#include "convolve.h"
#include "gradient.h"
#include "math/random.h"
#include "tensor/numeric.h"

namespace nano
{
        template <typename tiodata, typename tgdata>
        static void warp_by_field(tiodata&& iodata,
                const scalar_t alphax, const matrix_t& fieldx, const tgdata& gradx,
                const scalar_t alphay, const matrix_t& fieldy, const tgdata& grady,
                const scalar_t betaxy)
        {
                iodata.array() +=
                        alphax * fieldx.array() * gradx.array() +
                        alphay * fieldy.array() * grady.array() +
                        betaxy * (gradx.array().square() + grady.array().square()).sqrt();
        }

        template <typename trng>
        static void make_random_fields(const scalar_t noise,
                matrix_t& fieldx, matrix_t& fieldy, trng&& rng)
        {
                auto udist = make_udist<scalar_t>(-noise, +noise);
                nano::set_random(udist, rng, fieldx, fieldy);
        }

        template <typename trng>
        static void make_translation_fields(const scalar_t noise, const scalar_t delta,
                matrix_t& fieldx, matrix_t& fieldy, trng&& rng)
        {
                make_random_fields(noise, fieldx, fieldy, rng);
                fieldx.array() += delta;
                fieldy.array() += delta;
        }

        template <typename trng>
        static void make_rotation_fields(const scalar_t noise, const scalar_t delta, const scalar_t theta,
                matrix_t& fieldx, matrix_t& fieldy, trng&& rng)
        {
                static constexpr auto one = scalar_t(1);
                static constexpr auto half = scalar_t(0.5);

                const auto rows = fieldx.rows(), cols = fieldx.cols();
                const auto cx = half * static_cast<scalar_t>(cols);
                const auto cy = half * static_cast<scalar_t>(rows);
                const auto id = one / (nano::square(cx) + nano::square(cy));
                const auto cos_theta = std::cos(theta);
                const auto sin_theta = std::sin(theta);

                auto udist = make_udist<scalar_t>(-noise, +noise);

                for (tensor_size_t r = 0; r < rows; ++ r)
                {
                        for (tensor_size_t c = 0; c < cols; ++ c)
                        {
                                const auto dist = nano::square(scalar_t(r) - cy) + nano::square(scalar_t(c) - cx);

                                fieldx(r, c) = id * dist * cos_theta + udist(rng) + delta;
                                fieldy(r, c) = id * dist * sin_theta + udist(rng) + delta;
                        }
                }
        }

        template <typename ttensor, typename trng>
        void warp3d(ttensor&& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta, trng&& rng)
        {
                static constexpr auto one = scalar_t(1);
                static const auto pi = 4 * std::atan(one);

                const auto imaps = iodata.template size<0>();
                const auto irows = iodata.template size<1>();
                const auto icols = iodata.template size<2>();

                // generate random fields
                auto udist = make_udist<scalar_t>(-1, +1);
                const auto delta = udist(rng) * one;
                const auto theta = udist(rng) * pi / 8;

                matrix_t fieldx(irows, icols);
                matrix_t fieldy(irows, icols);
                switch (wtype)
                {
                case warp_type::translation:
                        make_translation_fields(noise, delta, fieldx, fieldy, rng);
                        break;

                case warp_type::rotation:
                        make_rotation_fields(noise, 0, theta, fieldx, fieldy, rng);
                        break;

                case warp_type::mixed:
                        make_rotation_fields(noise, delta, theta, fieldx, fieldy, rng);
                        break;

                case warp_type::random:
                default:
                        make_random_fields(noise, fieldx, fieldy, rng);
                        break;
                }

                // smooth fields
                const auto gauss = make_gauss_kernel(sigma);
                nano::convolve(gauss, fieldx);
                nano::convolve(gauss, fieldy);

                // mix input image with field-weighted gradients
                const auto alphax = udist(rng) * alpha;
                const auto alphay = udist(rng) * alpha;
                const auto betamx = udist(rng) * beta;

                for (auto d = 0; d < imaps; ++ d)
                {
                        const auto gradx = nano::gradientx(iodata.matrix(d));
                        const auto grady = nano::gradienty(iodata.matrix(d));

                        warp_by_field(iodata.matrix(d), alphax, fieldx, gradx, alphay, fieldy, grady, betamx);
                }
        }

        template <typename ttensor>
        void warp4d(ttensor&& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta)
        {
                for (auto plane = 0; plane < iodata.template size<0>(); ++ plane)
                {
                        warp3d(iodata.tensor(plane), wtype, noise, sigma, alpha, beta, make_rng());
                }
        }

        void warp(tensor3d_t& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta)
        {
                warp3d(iodata, wtype, noise, sigma, alpha, beta, make_rng());
        }

        void warp(tensor4d_t& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta)
        {
                warp4d(iodata, wtype, noise, sigma, alpha, beta);
        }

        void warp(tensor4d_map_t&& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta)
        {
                warp4d(iodata, wtype, noise, sigma, alpha, beta);
        }
}
