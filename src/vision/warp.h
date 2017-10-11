#pragma once

#include "tensor.h"
#include "math/random.h"
#include "vision/gauss.h"
#include "tensor/numeric.h"
#include "vision/convolve.h"
#include "vision/gradient.h"
#include "text/enum_string.h"

namespace nano
{
        ///
        /// \brief image warping type.
        ///
        enum class warp_type
        {
                translation,            ///<
                rotation,               ///<
                random,                 ///<
                mixed,                  ///< combines translation & rotation
        };

        template <>
        inline std::map<warp_type, std::string> enum_string<warp_type>()
        {
                return
                {
                        { warp_type::translation,       "translation" },
                        { warp_type::rotation,          "rotation" },
                        { warp_type::random,            "random" },
                        { warp_type::mixed,             "mixed" }
                };
        }

        ///
        /// \brief randomly warp images, like described in:
        ///      "Training Invariant Support Vector Machines using Selective Sampling", by
        ///      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        template <typename ttensor>
        void warp(ttensor& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta)
        {
                warp(iodata, wtype, noise, sigma, alpha, beta, make_rng<scalar_t>(-1, +1));
        }

        template <typename ttensor, typename trng>
        void warp(ttensor& iodata, const warp_type wtype,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta, trng&& rng)
        {
                static constexpr auto one = scalar_t(1);
                static constexpr auto half = scalar_t(0.5);
                static const auto pi = 4 * std::atan(one);

                const auto op_make_random_fields = [&] (matrix_t& fieldx, matrix_t& fieldy)
                {
                        nano::set_random([&] () { return rng() * noise; }, fieldx, fieldy);
                };

                const auto op_make_translation_fields = [&] (const scalar_t delta, matrix_t& fieldx, matrix_t& fieldy)
                {
                        nano::set_random([&] () { return rng() * noise + delta; }, fieldx, fieldy);
                };

                const auto op_make_rotation_fields = [&] (const scalar_t delta, const scalar_t theta, matrix_t& fieldx, matrix_t& fieldy)
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
                };

                const auto op_warp_by_field = [&] (auto&& iodata,
                        const scalar_t alphax, const matrix_t& fieldx, const auto& gradx,
                        const scalar_t alphay, const matrix_t& fieldy, const auto& grady,
                        const scalar_t beta)
                {
                        iodata.array() +=
                                alphax * fieldx.array() * gradx.array() +
                                alphay * fieldy.array() * grady.array() +
                                beta * (gradx.array().square() + grady.array().square()).sqrt();
                };

                const auto imaps = iodata.template size<0>();
                const auto irows = iodata.template size<1>();
                const auto icols = iodata.template size<2>();

                // generate random fields
                const auto delta = rng() * one;
                const auto theta = rng() * pi / 8;

                matrix_t fieldx(irows, icols);
                matrix_t fieldy(irows, icols);
                switch (wtype)
                {
                case warp_type::translation:
                        op_make_translation_fields(delta, fieldx, fieldy);
                        break;

                case warp_type::rotation:
                        op_make_rotation_fields(0, theta, fieldx, fieldy);
                        break;

                case warp_type::mixed:
                        op_make_rotation_fields(delta, theta, fieldx, fieldy);
                        break;

                case warp_type::random:
                default:
                        op_make_random_fields(fieldx, fieldy);
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

                        op_warp_by_field(iodata.matrix(d), alphax, fieldx, gradx, alphay, fieldy, grady, betamx);
                }
        }
}
