#pragma once

#include "layer.h"
#include "chrono/probe.h"

namespace nano
{
        class cnode_t;
        using cnodes_t = std::vector<cnode_t>;

        ///
        /// \brief computation node.
        ///
        class cnode_t
        {
        public:

                ///
                /// \brief constructor
                ///
                cnode_t(string_t name, string_t type, rlayer_t&& node) :
                        m_name(std::move(name)),
                        m_type(std::move(type)),
                        m_node(std::move(node))
                {
                }

                ///
                /// \brief copy constructor
                ///
                cnode_t(const cnode_t& other) :
                        m_name(other.m_name),
                        m_type(other.m_type),
                        m_node(other.m_node->clone()),
                        m_inodes(other.m_inodes),
                        m_onodes(other.m_onodes)
                {
                }

                ///
                /// \brief defaults
                ///
                cnode_t() = default;
                cnode_t(cnode_t&&) = default;
                cnode_t& operator=(cnode_t&&) = default;
                cnode_t& operator=(const cnode_t&) = delete;

                template <typename tvector>
                auto pdata(tvector&& buffer) const
                {
                        assert(m_pbegin >= 0 && m_pbegin + m_node->psize() <= buffer.size());
                        return map_vector(buffer.data() + m_pbegin, m_node->psize());
                }

                template <typename tvector>
                static auto idata(tvector&& buffer, const tensor_size_t count, const tensor3d_dim_t& idims)
                {
                        assert(count * nano::size(idims) <= buffer.size());
                        return map_tensor(buffer.data(), cat_dims(count, idims));
                }

                template <typename tvector>
                auto idata(tvector&& buffer, const tensor_size_t count,
                        const cnodes_t& nodes, const tensor3d_dim_t& idims) const
                {
                        std::vector<decltype(map_tensor(buffer.data(), make_dims(0, 0, 0, 0)))> imaps;
                        if (m_inodes.empty())
                        {
                                imaps.push_back(idata(buffer, count, idims));
                        }
                        else
                        {
                                for (const auto inode : m_inodes)
                                {
                                        imaps.push_back(nodes[inode].odata(buffer, count));
                                }
                        }
                        assert(!imaps.empty());
                        return imaps;
                }

                template <typename tvector>
                auto odata(tvector&& buffer, const tensor_size_t count) const
                {
                        assert(m_obegin >= 0 && m_obegin + count * osize(m_node) <= buffer.size());
                        return map_tensor(buffer.data() + m_obegin, cat_dims(count, m_node->odims()));
                }

                ///
                /// \brief compute the output (given the input & the parameters)
                ///
                void output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata)
                {
                        m_probe_output.measure([=] () { m_node->output(idata, pdata, odata); }, odata.size<0>());
                }

                ///
                /// \brief compute the gradient wrt the inputs (given the output & the parameters)
                ///
                void ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata)
                {
                        m_probe_ginput.measure([=] () { m_node->ginput(idata, pdata, odata); }, odata.size<0>());
                }

                ///
                /// \brief compute the (cumulated) gradient wrt the parameters (given the output & the input)
                ///
                void gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata)
                {
                        m_probe_gparam.measure([=] () { m_node->gparam(idata, pdata, odata); }, odata.size<0>());
                }

                // attributes
                string_t        m_name;
                string_t        m_type;
                rlayer_t        m_node;         ///< the computation node
                indices_t       m_inodes;       ///< input nodes
                indices_t       m_onodes;       ///< output nodes
                tensor_size_t   m_obegin{0};    ///< offset of the output tensor
                tensor_size_t   m_pbegin{0};    ///< offset of the parameter vector
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
