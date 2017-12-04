#pragma once

#include "layer.h"

namespace nano
{
        class cnode_t;
        using cnodes_t = std::vector<cnode_t>;

        ///
        /// \brief verify if the computation nodes form a valid graph:
        ///     - no cycles
        ///     - exactly one input
        ///     - exactly one output (for now)
        ///
        /// NB: if all the conditions are met, the computation nodes are sorted topologically.
        ///
        bool check_nodes(cnodes_t&);

        ///
        /// \brief computation node.
        ///
        class cnode_t
        {
        public:

                cnode_t() = default;
                cnode_t(const cnode_t&);
                cnode_t(cnode_t&&) = default;
                cnode_t& operator=(const cnode_t&);
                cnode_t& operator=(cnode_t&&) = default;
                cnode_t(string_t name, string_t type, rlayer_t&&);

                template <typename tvector>
                auto pdata(tvector&& buffer) const
                {
                        assert(m_pbegin >= 0 && m_pbegin + m_node->psize() <= buffer.size());
                        return map_vector(buffer.data() + m_pbegin, m_node->psize());
                }

                template <typename tvector>
                auto idata(tvector&& buffer, const tensor_size_t count) const
                {
                        // todo: handle multiple inputs
                        assert(m_ibegin >= 0 && m_ibegin + count * m_node->isize() <= buffer.size());
                        return map_tensor(buffer.data() + m_ibegin, cat_dims(count, m_node->idims()));
                }

                template <typename tvector>
                auto odata(tvector&& buffer, const tensor_size_t count) const
                {
                        assert(m_obegin >= 0 && m_obegin + count * m_node->osize() <= buffer.size());
                        return map_tensor(buffer.data() + m_obegin, cat_dims(count, m_node->odims()));
                }

                // attributes
                string_t        m_name;
                string_t        m_type;
                rlayer_t        m_node;         ///< the computation node
                indices_t       m_inodes;       ///< input nodes
                indices_t       m_onodes;       ///< output nodes
                tensor_size_t   m_ibegin{0};    ///< offset of the input tensor
                tensor_size_t   m_obegin{0};    ///< offset of the output tensor
                tensor_size_t   m_pbegin{0};    ///< offset of the parameter vector
        };
}
