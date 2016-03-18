#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// \brief task containing random fixed-size annotated 3D input tensors.
        ///
        class NANO_PUBLIC random_task_t : public mem_tensor_task_t
        {
        public:

                NANO_MAKE_CLONABLE(random_task_t,
                        "random tasks: idims=10[1,100],irows=32[1,100],icols=32[1,100],osize=10[1,100],"\
                        "count=1024[256,1M],folds=1[1,10]")

                ///
                /// \brief constructor
                ///
                explicit random_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate(const string_t& dir) override;

        private:

                // attributes
                size_t          m_count;
        };
}
