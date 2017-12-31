#pragma once

#include "enhancer.h"

namespace nano
{
        ///
        /// \brief generate samples by adding random samples without a label/class.
        ///
        /// NB: the idea is that natural images are just a small subset of all possible images, so most likely
        ///     a random image will have no label/class, so its target is {-1}^C (C = number of classes).
        ///
        class enhancer_noclass_t final : public enhancer_t
        {
        public:

                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;
                minibatch_t get(const task_t&, const fold_t&, const size_t begin, const size_t end) const final;

        private:

                // attributes
                scalar_t        m_ratio{static_cast<scalar_t>(0.1)};
                scalar_t        m_noise{static_cast<scalar_t>(0.1)};
        };
}
