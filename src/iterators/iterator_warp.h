#pragma once

#include "iterator.h"

namespace nano
{
        ///
        /// \brief generate samples by randomly warp images, like described in:
        ///      "Training Invariant Support Vector Machines using Selective Sampling", by
        ///      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        struct iterator_warp_t final : public iterator_t
        {
                explicit iterator_warp_t(const string_t& configuration = string_t());

                virtual sample_t get(const task_t&, const fold_t&, const size_t index) const final;
        };
}
