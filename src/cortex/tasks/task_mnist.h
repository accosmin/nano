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
        class mnist_task_t : public mem_vision_task_t
        {
        public:

                NANO_MAKE_CLONABLE(mnist_task_t, "MNIST (digit classification)")

                ///
                /// \brief constructor
                ///
                explicit mnist_task_t(const string_t& configuration = string_t());

       private:

                virtual bool populate(const string_t& dir) override;

                // load binary file
                bool load_binary(const string_t& ifile, const string_t& gfile, const protocol, const size_t count);
        };
}
