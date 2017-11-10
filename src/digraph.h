#pragma once

#include <vector>
#include <cassert>
#include <cstdint>
#include <algorithm>

namespace nano
{
        ///
        /// \brief generic directed graph specified by a set of edges.
        ///
        template <typename tindex>
        class digraph_t
        {
        public:
                using edge_t = std::pair<tindex, tindex>;
                using edges_t = std::vector<edge_t>;
                using indices_t = std::vector<tindex>;

                ///
                /// \brief
                ///
                digraph_t() = default;

                ///
                /// \brief initialize the digraph with the given number of vertices
                ///
                digraph_t(const tindex vertices) : m_vertices(vertices) {}

                ///
                /// \brief create a directed edge between the src and the dst vertex ids
                /// \return true if the vertex ids are valid
                ///
                bool edge(const tindex src, const tindex dst);

                ///
                /// \brief returns the vertices with no incoming edge
                ///
                indices_t incoming() const;

                ///
                /// \brief returns the incoming vertices of the given vertex id
                ///
                indices_t incoming(const tindex dst) const;

                ///
                /// \brief returns the vertices with no outgoing edge
                ///
                indices_t outgoing() const;

                ///
                /// \brief returns the outgoing vertices of the given vertex id
                ///
                indices_t outgoing(const tindex src) const;

                ///
                /// \brief depth-first search where the given operator is called with the current vertex id
                /// \return true if the graph is not a DAG
                ///
                template <typename tvcall>
                bool depth_first(const tvcall&) const;

                ///
                /// \brief checks if the graph is a DAG
                ///
                bool dag() const;

                ///
                /// \brief topologically sort the graph
                /// \return the list of sorted vertices
                ///
                indices_t tsort() const;

                ///
                /// \brief access functions
                ///
                auto vertices() const { return m_vertices; }
                const auto& edges() const { return m_edges; }
                auto empty() const { return m_edges.empty() && !m_vertices; }

        private:

                enum class flag : uint8_t
                {
                        none,
                        temporary,
                        permanent,
                };

                template <typename tvcall>
                bool visit(std::vector<flag>& flags, const tvcall& vcall, const tindex vindex) const
                {
                        assert(flags.size() == vertices());

                        switch (flags[vindex])
                        {
                        case flag::temporary:
                                return false;   // !DAG

                        case flag::permanent:
                                return true;

                        case flag::none:
                        default:
                                flags[vindex] = flag::temporary;
                                for (const auto& edge : m_edges)
                                {
                                        if (edge.first == vindex && !visit(flags, vcall, edge.second))
                                        {
                                                return false;
                                        }
                                }
                                flags[vindex] = flag::permanent;
                                vcall(vindex);
                                return true;
                        }
                }

                static void unique(indices_t& set)
                {
                        std::sort(set.begin(), set.end());
                        set.erase(std::unique(set.begin(), set.end()), set.end());
                }

                static indices_t difference(indices_t& set1, indices_t& set2)
                {
                        unique(set1);
                        unique(set2);

                        indices_t diff;
                        std::set_difference(
                                set1.begin(), set1.end(), set2.begin(), set2.end(),
                                std::inserter(diff, diff.begin()));
                        return diff;
                }

                // attributes
                edges_t         m_edges;
                tindex          m_vertices{0};
        };

        template <typename tindex>
        bool digraph_t<tindex>::edge(const tindex src, const tindex dst)
        {
                if (    src < vertices() && dst < vertices() &&
                        std::find_if(m_edges.begin(), m_edges.end(),
                        [=] (const auto& edge) { return edge.first == src && edge.second == dst; }) == m_edges.end())
                {
                        m_edges.emplace_back(src, dst);
                        return true;
                }
                else
                {
                        return false;
                }
        }

        template <typename tindex>
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::incoming() const
        {
                indices_t srcs, dsts;
                for (const auto& edge : m_edges)
                {
                        srcs.emplace_back(edge.first);
                        dsts.emplace_back(edge.second);
                }

                return difference(srcs, dsts);
        }

        template <typename tindex>
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::incoming(const tindex dst) const
        {
                indices_t srcs;
                for (const auto& edge : m_edges)
                {
                        if (edge.second == dst)
                        {
                                srcs.emplace_back(edge.first);
                        }
                }

                unique(srcs);
                return srcs;
        }

        template <typename tindex>
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::outgoing() const
        {
                indices_t srcs, dsts;
                for (const auto& edge : m_edges)
                {
                        srcs.emplace_back(edge.first);
                        dsts.emplace_back(edge.second);
                }

                return difference(dsts, srcs);
        }

        template <typename tindex>
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::outgoing(const tindex src) const
        {
                indices_t dsts;
                for (const auto& edge : m_edges)
                {
                        if (src == edge.first)
                        {
                                dsts.emplace_back(edge.second);
                        }
                }

                unique(dsts);
                return dsts;
        }

        template <typename tindex>
        template <typename tvcall>
        bool digraph_t<tindex>::depth_first(const tvcall& vcall) const
        {
                std::vector<flag> flags(vertices(), flag::none);

                // start from the vertices that don't have incoming edges
                for (const auto src : incoming())
                {
                        if (!visit(flags, vcall, src))
                        {
                                return false;
                        }
                }

                // check if any remaning not visited vertices (e.g. another cycle in the graph not connected)
                for (tindex src = 0; src < flags.size(); ++ src)
                {
                        if (flags[src] == flag::none && !visit(flags, vcall, src))
                        {
                                return false;
                        }
                }
                return true;
        }

        template <typename tindex>
        bool digraph_t<tindex>::dag() const
        {
                return depth_first([] (const tindex vindex) { (void)vindex; });
        }

        template <typename tindex>
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::tsort() const
        {
                indices_t vindices;
                depth_first([&] (const tindex vindex) { vindices.push_back(vindex); });

                if (vindices.size() == vertices())
                {
                        std::reverse(vindices.begin(), vindices.end());
                }
                else
                {
                        vindices.clear();       // !DAG
                }
                return vindices;
        }
}
