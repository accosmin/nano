#pragma once

#include "task_mem.hpp"

namespace nano
{
        struct mem_tensor_sample_t
        {
                explicit mem_tensor_sample_t(
                        const size_t index = 0,
                        const vector_t& target = vector_t(),
                        const string_t& label = string_t()) :
                        m_index(index), m_target(target), m_label(label) {}

                auto index() const { return m_index; }
                auto input(const tensor3d_t& tensor) const { return tensor; }
                auto target() const { return m_target; }
                auto label() const { return m_label; }

                // attributes
                size_t          m_index;
                vector_t        m_target;
                string_t        m_label;
        };

        ///
        /// \brief in-memory generic task consisting of generic 3D input tensors.
        ///
        class NANO_PUBLIC mem_tensor_task_t : public mem_task_t<tensor3d_t, mem_tensor_sample_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                mem_tensor_task_t(
                        const string_t& name,
                        const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t osize,
                        const size_t fsize) :
                        mem_task_t<tensor3d_t, mem_tensor_sample_t>(name, idims, irows, icols, osize, fsize) {}

                ///
                /// \brief destructor
                ///
                virtual ~mem_tensor_task_t() {}
        };
}
