#pragma once

#include "io/archive.h"
#include "task_mem_vision.h"

namespace nano
{
        ///
        /// CIFAR10 task:
        ///      - object classification
        ///      - 32x32 color images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://www.cs.toronto.edu/~kriz/cifar.html
        ///
        class cifar10_task_t final : public mem_vision_task_t
        {
        public:

                cifar10_task_t();
                bool populate() override;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

        private:

                bool load_binary(const string_t& filename, istream_t&, const protocol, const size_t);

                // attributes
                string_t        m_dir;  ///< directory where to load the task from
        };
}
