#pragma once

#include "tensor.h"
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
        void warp(tensor3d_t&, const warp_type,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta);

        void warp(tensor3d_t&, const warp_type,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta,
                matrix_t& fieldx, matrix_t& fieldy, matrix_t& gradx, matrix_t& grady);
}
