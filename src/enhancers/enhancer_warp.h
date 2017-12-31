#pragma once

#include "enhancer.h"
#include "vision/warp.h"

namespace nano
{
        ///
        /// \brief generate samples by randomly warp images, like described in:
        ///      "Training Invariant Support Vector Machines using Selective Sampling", by
        ///      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        class enhancer_warp_t final : public enhancer_t
        {
        public:

                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;
                minibatch_t get(const task_t&, const fold_t&, const size_t begin, const size_t end) const final;

        private:

                // attributes
                warp_type       m_type{warp_type::mixed};
                scalar_t        m_noise{static_cast<scalar_t>(0.1)};
                scalar_t        m_sigma{4};
                scalar_t        m_alpha{1};
                scalar_t        m_beta{1};
        };
}
