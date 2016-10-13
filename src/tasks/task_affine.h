#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// \brief task containing affine-mapped 3D input tensors: f(x) = A * x + b + noise.
        ///
        class NANO_PUBLIC affine_task_t : public mem_tensor_task_t
        {
        public:

                NANO_MAKE_CLONABLE(affine_task_t)

                ///
                /// \brief constructor
                ///
                explicit affine_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate() override final;

        private:

                // attributes
                size_t          m_count;
                scalar_t        m_noise;
        };
}
