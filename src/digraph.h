#pragma once

#include <vector>
#include <cassert>
#include <cstdint>
#include <algorithm>

namespace nano
{
        ///
        /// \brief generic directed graph specified by a set of edges.
        ///     todo: keep the edges sorted by the first index
        ///     todo: use binary search to find the incoming (outgoing) vertices to a given vertex
        ///
        template <typename tindex>
        class digraph_t
        {
        public:
                using edge_t = std::pair<tindex, tindex>;
                using edges_t = std::vector<edge_t>;
                using indices_t = std::vector<tindex>;

                ///
                /// \brief create a directed edge between the src and the dst vertex ids
                ///
                void edge(const tindex src, const tindex dst);

                ///
                /// \brief remove all edges
                ///
                void clear();

                ///
                /// \brief prepares internals once all the edges have been added
                ///
                void done();

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

                template <typename toperator>
                auto foreach_outgoing(const tindex src, const toperator& top) const
                {
                        const auto sedge = edge_t{src, src};
                        const auto ecomp = [] (const auto& e1, const auto& e2) { return e1.first < e2.first; };
                        const auto range = std::equal_range(m_edges.begin(), m_edges.end(), sedge, ecomp);

                        return std::find_if(range.first, range.second, top) == range.second;
                }

                template <typename tvcall>
                bool visit(std::vector<flag>& flags, const tvcall& vcall, const tindex src) const
                {
                        assert(flags.size() == vertices());

                        switch (flags[src])
                        {
                        case flag::temporary:
                                return false;   // !DAG

                        case flag::permanent:
                                return true;

                        case flag::none:
                        default:
                                flags[src] = flag::temporary;
                                if (!foreach_outgoing(src, [&] (const auto& e) { return !visit(flags, vcall, e.second); }))
                                {
                                        return false;
                                }
                                flags[src] = flag::permanent;
                                vcall(src);
                                return true;
                        }
                }

                template <typename tvalue>
                static void unique(std::vector<tvalue>& set)
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
                edges_t         m_edges;        ///<
                tindex          m_vertices{0};  ///< total number of vertices (maximum edge index + 1)
        };

        template <typename tindex>
        void digraph_t<tindex>::clear()
        {
                m_edges.clear();
                m_vertices = 0;
        }

        template <typename tindex>
        void digraph_t<tindex>::edge(const tindex src, const tindex dst)
        {
                m_edges.emplace_back(src, dst);
                m_vertices = std::max(m_vertices, static_cast<tindex>(src + 1));
                m_vertices = std::max(m_vertices, static_cast<tindex>(dst + 1));
        }

        template <typename tindex>
        void digraph_t<tindex>::done()
        {
                unique(m_edges);
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
                foreach_outgoing(src, [&] (const auto& edge) { dsts.emplace_back(edge.second); return true; });

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
