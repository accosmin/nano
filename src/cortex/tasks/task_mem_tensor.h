#pragma once

#include "task_mem.h"

namespace nano
{
        namespace detail
        {
                struct mem_tensor_sample_t
                {
                        const tensor3d_t& input() const { return m_input; }
                        const target_t& target() const { return m_target; }

                        tensor3d_t      m_input;
                        target_t        m_target;
                };
        }

        ///
        /// \brief in-memory generic task consisting of generic 3D input tensors.
        ///
        class NANO_PUBLIC mem_tensor_task_t : public mem_task_t<detail::mem_tensor_sample_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                mem_tensor_task_t(
                        const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t osize) :
                        mem_task_t<detail::mem_tensor_sample_t>(idims, irows, icols, osize) {}

                ///
                /// \brief destructor
                ///
                virtual ~mem_tensor_task_t() {}
        };
}
