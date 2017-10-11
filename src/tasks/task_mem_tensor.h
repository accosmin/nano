#pragma once

#include "task_mem.h"
#include "math/hash.h"

namespace nano
{
        struct mem_tensor_sample_t
        {
                explicit mem_tensor_sample_t(
                        const size_t index = 0,
                        const tensor3d_t& target = tensor3d_t(),
                        const string_t& label = string_t()) :
                        m_index(index), m_target(target), m_label(label)
                {
                }

                mem_tensor_sample_t(
                        const size_t index,
                        const vector_t& target,
                        const string_t& label = string_t()) :
                        mem_tensor_sample_t(index, nano::map_tensor(target.data(), target.size(), 1, 1), label)
                {
                }

                auto index() const { return m_index; }
                auto input(const tensor3d_t& tensor) const { return tensor; }
                auto ihash(const size_t seed) const { return seed; }
                auto ohash() const { return nano::hash_range(m_target.data(), m_target.data() + m_target.size()); }
                auto output() const { return m_target; }
                auto label() const { return m_label; }

                // attributes
                size_t          m_index;        ///< input tensor index
                tensor3d_t      m_target;       ///<
                string_t        m_label;        ///<
        };

        ///
        /// \brief in-memory generic task consisting of generic 3D input tensors.
        ///
        using mem_tensor_task_t = mem_task_t<tensor3d_t, mem_tensor_sample_t>;
}
