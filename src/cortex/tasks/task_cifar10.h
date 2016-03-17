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

                NANO_MAKE_CLONABLE(cifar10_task_t, "CIFAR-10 (object classification)")

                ///
                /// \brief constructor
                ///
                explicit cifar10_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate(const string_t& dir) override;

                // load binary file
                bool load(const string_t& filename, const char* bdata, size_t bdata_size, protocol p, size_t count);
        };
}

