#pragma once

#include "arch.h"
#include "tensor.h"

namespace cortex
{
        enum class field_type
        {
                translation,
                rotation,
                random,
        };

        ///
        /// \brief parameters describing a random warping, like described in
        //      "Training Invariant Support Vector Machines using Selective Sampling", by
        //      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        struct NANOCV_PUBLIC warp_params
        {
                explicit warp_params(
                        field_type ftype = field_type::random,
                        scalar_t noise = 0.1,
                        scalar_t sigma = 4.0,
                        scalar_t alpha = 1.0,
                        scalar_t beta = 1.0);

                field_type      m_ftype;
                scalar_t        m_noise;                ///< noise level of the fields
                scalar_t        m_sigma;                ///< standard deviation of the Gaussian to smooth the fields
                scalar_t        m_alpha;                ///< field mixing
                scalar_t        m_beta;                 ///< magnitude mixing
        };

        ///
        /// \brief randomly warp the input RGBA tensor
        ///
        NANOCV_PUBLIC tensor_t warp(const tensor_t& image, const warp_params& params, tensor_t* field_image = nullptr);
}
