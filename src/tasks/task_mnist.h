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

                explicit mnist_task_t(const string_t& configuration = string_t());

                virtual rtask_t clone() const override;

       private:

                virtual bool populate() override;

                // load binary file
                bool load_binary(const string_t& ifile, const string_t& gfile, const protocol, const size_t count);
        };
}
