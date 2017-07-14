#pragma once

#include "enhancer.h"

namespace nano
{
        ///
        /// \brief generate samples by randomly warp images, like described in:
        ///      "Training Invariant Support Vector Machines using Selective Sampling", by
        ///      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        struct enhancer_warp_t final : public enhancer_t
        {
                explicit enhancer_warp_t(const string_t& params = string_t());

                virtual sample_t get(const task_t&, const fold_t&, const size_t index) const final;
        };
}
