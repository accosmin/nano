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

                virtual tensor3d_t input(const size_t index) const final;
                virtual tensor3d_t target(const size_t index) const final;
        };
}
