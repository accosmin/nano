#pragma once

#include "enhancer.h"

namespace nano
{
        ///
        /// \brief use the samples as they are.
        ///
        class enhancer_default_t final : public enhancer_t
        {
        public:

                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;
                minibatch_t get(const task_t&, const fold_t&, const size_t begin, const size_t end) const final;
        };
}
