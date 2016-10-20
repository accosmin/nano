#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// \brief task containing affine-mapped 3D input tensors: f(x) = A * x + b + noise.
        ///
        class NANO_PUBLIC affine_task_t final : public mem_tensor_task_t
        {
        public:

                explicit affine_task_t(const string_t& configuration = string_t());

                virtual rtask_t clone(const string_t& configuration) const;
                virtual rtask_t clone() const;

        private:

                virtual bool populate() override;
        };
}
