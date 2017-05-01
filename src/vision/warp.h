#pragma once

#include "arch.h"
#include "tensor.h"
#include "math/random.h"

namespace nano
{
        enum class field_type
        {
                translation,
                rotation,
                random,
        };

        ///
        /// \brief utility to randomly warp images, like described in:
        ///      "Training Invariant Support Vector Machines using Selective Sampling", by
        ///      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        struct NANO_PUBLIC warper_t
        {
                ///
                /// \brief constructor
                ///
                explicit warper_t(
                        field_type ftype = field_type::random,
                        scalar_t noise = scalar_t(0.1),
                        scalar_t sigma = scalar_t(4.0),
                        scalar_t alpha = scalar_t(1.0),
                        scalar_t beta = scalar_t(1.0));

                ///
                /// \brief randomly warp the input tensor.
                /// NB: the warping is performed independently for each plane.
                ///
                void operator()(const tensor3d_t& input, tensor3d_t& output, tensor3d_t* field_image = nullptr);

        private:

                using rng_t = random_t<scalar_t>;

                // attributes
                field_type      m_ftype;
                scalar_t        m_noise;        ///< noise level of the fields
                scalar_t        m_sigma;        ///< standard deviation of the Gaussian to smooth the fields
                scalar_t        m_alpha;        ///< field mixing
                scalar_t        m_beta;         ///< magnitude mixing
                tensor3d_t      m_gradx;        ///< buffer: horizontal gradient per plane
                tensor3d_t      m_grady;        ///< buffer: vertical gradient per plane
                matrix_t        m_fieldx;       ///< buffer: horizontal displacement
                matrix_t        m_fieldy;       ///< buffer: vertical displacement
        };
}
