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

                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) const final;
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) const final;
        };
}
