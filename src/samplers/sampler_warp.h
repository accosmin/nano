#pragma once

#include "sampler.h"

namespace nano
{
        ///
        /// \brief generate samples by randomly warp images, like described in:
        ///      "Training Invariant Support Vector Machines using Selective Sampling", by
        ///      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        struct sampler_warp_t final : public sampler_t
        {
                explicit sampler_warp_t(const string_t& configuration = string_t());

                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) final;
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) final;

        private:

                // attributes
                matrix_t        m_gradx;        ///< buffer: horizontal gradient
                matrix_t        m_grady;        ///< buffer: vertical gradient
                matrix_t        m_fieldx;       ///< buffer: horizontal displacement
                matrix_t        m_fieldy;       ///< buffer: vertical displacement
        };
}
