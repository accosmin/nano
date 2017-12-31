#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// \brief synthetic task for the n-parity binary classification problem.
        ///     given n-dimensional {0,1} vectors, the task is to predict:
        ///             +1 if its has an odd number of ones
        ///             -1 otherwise
        ///
        /// parameters:
        ///     n       - dimension of the input vectors
        ///     count   - number of samples (training + validation + test)
        ///
        class nparity_task_t final : public mem_tensor_task_t
        {
        public:

                nparity_task_t();
                bool populate() override;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

        private:

                // attributes
                tensor_size_t   m_dims{32};
                size_t          m_count{1024};
        };
}
