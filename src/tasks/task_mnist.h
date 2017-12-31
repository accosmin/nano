#pragma once

#include "task_mem_vision.h"

namespace nano
{
        ///
        /// MNIST task:
        ///      - digit classification
        ///      - 28x28 grayscale images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://yann.lecun.com/exdb/mnist/
        ///
        class mnist_task_t final : public mem_vision_task_t
        {
        public:

                mnist_task_t();
                bool populate() override;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

        private:

                bool load_binary(const string_t& ifile, const string_t& gfile, const protocol, const size_t count);

                // attributes
                string_t        m_dir;  ///< directory where to load the task from
        };
}
