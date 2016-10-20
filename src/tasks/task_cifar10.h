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
        class cifar10_task_t final : public mem_vision_task_t
        {
        public:

                explicit cifar10_task_t(const string_t& configuration = string_t());

                virtual rtask_t clone(const string_t& configuration) const;
                virtual rtask_t clone() const;

        private:

                virtual bool populate() override;

                bool load_binary(const string_t& filename, const char*, const size_t, const protocol, const size_t);
        };
}

