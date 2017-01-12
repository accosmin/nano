#pragma once

#include "task_mem_vision.h"

namespace nano
{
        class archive_stream_t;

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

                explicit cifar10_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate() override;

                bool load_binary(const string_t& filename, archive_stream_t&, const protocol, const size_t);
        };
}

