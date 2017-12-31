#pragma once

#include "enhancer.h"

namespace nano
{
        ///
        /// \brief generate samples by adding salt & pepper noise to inputs.
        ///
        class enhancer_noise_t final : public enhancer_t
        {
        public:

                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;
                minibatch_t get(const task_t&, const fold_t&, const size_t begin, const size_t end) const final;

        private:

                // attributes
                scalar_t        m_noise{static_cast<scalar_t>(0.1)};
        };
}
