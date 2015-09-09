#pragma once

#include "nanocv/tensor.h"

namespace ncv
{
        enum class field_type
        {
                translation,
                rotation,
                random,
        };

        ///
        /// \brief parameters describing a random warping
        ///
        struct warp_params
        {
                explicit warp_params(
                        field_type ftype = field_type::random,
                        scalar_t noise = 0.1,
                        scalar_t sigma = 4.0)
                        :       m_ftype(ftype),
                                m_noise(noise),
                                m_sigma(sigma)
                {
                }

                field_type      m_ftype;
                scalar_t        m_noise;
                scalar_t        m_sigma;
        };

        ///
        /// \brief randomly warp the input RGBA tensor
        ///
        NANOCV_PUBLIC tensor_t warp(const tensor_t& image, const warp_params& params, tensor_t* field_image = nullptr);
}
