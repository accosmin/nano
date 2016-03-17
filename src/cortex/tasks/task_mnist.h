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

                ///
                /// \brief short name of this task
                ///
                virtual string_t name() const override { return "mnist"; }

                ///
                /// \brief load the task from the given directory (if possible)
                ///
                virtual bool load(const string_t& dir = string_t()) override;

       private:

                // load binary file
                bool load(const string_t& ifile, const string_t& gfile, protocol p, size_t count);
        };
}
