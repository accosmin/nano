#include "math/gauss.h"
#include "math/random.h"
#include "math/numeric.h"
#include "sampler_warp.h"
#include "tensor/numeric.h"
#include "text/to_params.h"
#include "vision/convolve.h"
#include "vision/gradient.h"
#include "text/from_params.h"

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

                for (tensor_size_t r = 0; r < rows; ++ r)
                {
                        for (tensor_size_t c = 0; c < cols; ++ c)
                        {
                                const auto dist = nano::square(scalar_t(r) - cy) + nano::square(scalar_t(c) - cx);

                                fieldx(r, c) = id * dist * std::cos(theta) + rng() * noise + delta;
                                fieldy(r, c) = id * dist * std::sin(theta) + rng() * noise + delta;
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

        static void warp(tensor3d_t& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta,
                matrix_t& fieldx, matrix_t& fieldy, matrix_t& gradx, matrix_t& grady)
        {
                const auto imaps = iodata.size<0>();
                const auto irows = iodata.size<1>();
                const auto icols = iodata.size<2>();

                auto rng = make_rng<scalar_t>(-1, +1);

                fieldx.resize(irows, icols);
                fieldy.resize(irows, icols);

                gradx.resize(irows, icols);
                grady.resize(irows, icols);

                // generate random fields
                const auto delta = rng() * one;
                const auto theta = rng() * pi / 8;

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
                const nano::gauss_kernel_t<scalar_t> gauss(sigma);
                nano::convolve(gauss, fieldx);
                nano::convolve(gauss, fieldy);

                // mix input image with field-weighted gradients
                const auto alphax = rng() * alpha;
                const auto alphay = rng() * alpha;
                const auto betamx = rng() * beta;

                for (auto d = 0; d < imaps; ++ d)
                {
                        nano::gradientx(iodata.matrix(d), gradx);
                        nano::gradienty(iodata.matrix(d), grady);

                        warp_by_field(iodata.matrix(d), alphax, fieldx, gradx, alphay, fieldy, grady, betamx);
                }
        }

        void warp(tensor3d_t& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta)
        {
                matrix_t fieldx, fieldy, gradx, grady;
                warp(iodata, wtype, noise, sigma, alpha, beta, fieldx, fieldy, gradx, grady);
        }

        sampler_warp_t::sampler_warp_t(const string_t& config) :
                sampler_t(to_params(config,
                "type", to_string(warp_type::mixed) + "[" + concatenate(enum_values<warp_type>(warp_type::mixed)) + "]",
                "noise", "0.1[0,1]",
                "sigma", "4.0[0,10]",
                "alpha", "1.0[0,10]",
                "beta", "1.0[0,10]"))
        {
        }

        tensor3d_t sampler_warp_t::input(const task_t& task, const fold_t& fold, const size_t index)
        {
                const auto wtype = from_params<warp_type>(config(), "type");
                const auto noise = from_params<scalar_t>(config(), "noise");
                const auto sigma = from_params<scalar_t>(config(), "sigma");
                const auto alpha = from_params<scalar_t>(config(), "alpha");
                const auto beta = from_params<scalar_t>(config(), "beta");

                tensor3d_t iodata = task.input(fold, index);
                warp(iodata, wtype, noise, sigma, alpha, beta, m_fieldx, m_fieldy, m_gradx, m_grady);
                return iodata;
        }

        tensor3d_t sampler_warp_t::target(const task_t& task, const fold_t& fold, const size_t index)
        {
                return task.target(fold, index);
        }
}
