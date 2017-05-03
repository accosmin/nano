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
        struct mnist_task_t final : public mem_vision_task_t
        {
                explicit mnist_task_t(const string_t& configuration = string_t());

                virtual bool populate() override;

                bool load_binary(const string_t& ifile, const string_t& gfile, const protocol, const size_t count);
        };
}
