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
        struct cifar10_task_t final : public mem_vision_task_t
        {
                explicit cifar10_task_t(const string_t& params = string_t());

                virtual bool populate() override;

                bool load_binary(const string_t& filename, istream_t&, const protocol, const size_t);
        };
}
