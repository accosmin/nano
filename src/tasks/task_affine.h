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

                NANO_MAKE_CLONABLE(affine_task_t,
                        "synthetic affine task",
                        "idims=10[1,100],irows=32[1,100],icols=32[1,100],osize=10[1,1000],"\
                        "count=1000[10,1M],noise=0.1[0.001,0.5]")

                ///
                /// \brief constructor
                ///
                explicit affine_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate() override;

        private:

                // attributes
                size_t          m_count;
                scalar_t        m_noise;
        };
}
