#pragma once

#include "tensor.h"
#include "text/cast.h"

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
        inline enum_map_t<warp_type> enum_string<warp_type>()
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
        void warp(tensor3d_t& iodata, const warp_type,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta);

        void warp(tensor4d_t& iodata, const warp_type,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta);

        void warp(tensor4d_map_t&& iodata, const warp_type,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta);
}
