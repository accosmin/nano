#pragma once

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
        class cifar10_task_t : public mem_vision_task_t
        {
        public:

                NANO_MAKE_CLONABLE(cifar10_task_t)

                ///
                /// \brief constructor
                ///
                explicit cifar10_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate() final;

                // load binary file
                bool load_binary(const string_t& filename, const char*, const size_t, const protocol, const size_t);
        };
}

