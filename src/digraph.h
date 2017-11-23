#pragma once

#include <deque>
#include <stack>
#include <vector>
#include <cassert>
#include <cstdint>
#include <algorithm>

namespace nano
{
        ///
        /// \brief generic directed graph specified by a set of edges.
        ///     todo: faster (aka lower complexity) for ::sink & ::sinks
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
                /// \brief returns the source vertices (aka the vertices without incoming vertices)
                ///
                indices_t sources() const;

                ///
                /// \brief returns the incoming vertices of the given vertex id
                ///
                indices_t in(const tindex dst) const;

                ///
                /// \brief returns the sink vertices (aka the vertices without outgoing vertices)
                ///
                indices_t sinks() const;

                ///
                /// \brief returns the outgoing vertices of the given vertex id
                ///
                indices_t out(const tindex src) const;

                ///
                /// \brief depth-first search where the given operator is called with the current vertex id
                /// \return true if the graph is not a DAG
                ///
                template <typename tvcall>
                bool depth_first(const tvcall&) const;

                ///
                /// \brief breadth-first search where the given operator is called with the current vertex id
                /// \return true if the graph is not a DAG
                ///
                template <typename tvcall>
                bool breadth_first(const tvcall&) const;

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
                /// \brief check if there is a path between the source and the destination vertices
                ///
                bool connected(const tindex src, const tindex dst) const;

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
                        visited,
                };

                using flags_t = std::vector<flag>;

                template <typename toperator>
                bool foreach_out(const tindex src, const toperator& top) const
                {
                        const auto sedge = edge_t{src, src};
                        const auto ecomp = [] (const auto& e1, const auto& e2) { return e1.first < e2.first; };
                        const auto range = std::equal_range(m_edges.begin(), m_edges.end(), sedge, ecomp);

                        return std::find_if_not(range.first, range.second, top) == range.second;
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

                template <typename tvcall, typename tqueuer>
                static bool visit(flags_t& flags, const tvcall& vcall, const tqueuer& q, const tindex v)
                {
                        assert(v < flags.size());
                        switch (flags[v])
                        {
                        case flag::none:
                                q();
                                flags[v] = flag::visited;
                                vcall(v);
                                return true;
                        default:
                                return false;
                        }
                }

                template <typename tvcall>
                static bool depth_visit(flags_t& flags, const tvcall& vcall, std::stack<tindex>& q, const tindex v)
                {
                        return visit(flags, vcall, [&] () { q.push(v); }, v);
                }

                template <typename tvcall>
                static bool breadth_visit(flags_t& flags, const tvcall& vcall, std::deque<tindex>& q, const tindex v)
                {
                        return visit(flags, vcall, [&] () { q.push_back(v); }, v);
                }

                template <typename tvcall>
                bool depth_first(flags_t& flags, const tvcall& vcall, const tindex src) const
                {
                        std::stack<tindex> stack;
                        if (!depth_visit(flags, vcall, stack, src))
                        {
                                return false;
                        }

                        while (!stack.empty())
                        {
                                const auto u = stack.top();
                                stack.pop();

                                if (!foreach_out(u, [&] (const edge_t& e) { return depth_visit(flags, vcall, stack, e.second); }))
                                {
                                        return false;
                                }
                        }
                        return true;
                }

                template <typename tvcall>
                bool breadth_first(flags_t& flags, const tvcall& vcall, const tindex src) const
                {
                        std::deque<tindex> queue;
                        if (!breadth_visit(flags, vcall, queue, src))
                        {
                                return false;
                        }

                        while (!queue.empty())
                        {
                                const auto u = queue.front();
                                queue.pop_front();

                                if (!foreach_out(u, [&] (const edge_t& e) { return breadth_visit(flags, vcall, queue, e.second); }))
                                {
                                        return false;
                                }
                        }
                        return true;
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
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::sources() const
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
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::in(const tindex dst) const
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
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::sinks() const
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
        typename digraph_t<tindex>::indices_t digraph_t<tindex>::out(const tindex src) const
        {
                indices_t dsts;
                foreach_out(src, [&] (const auto& edge) { dsts.emplace_back(edge.second); return true; });

                unique(dsts);
                return dsts;
        }

        template <typename tindex>
        template <typename tvcall>
        bool digraph_t<tindex>::depth_first(const tvcall& vcall) const
        {
                flags_t flags(vertices(), flag::none);
                for (const auto src : sources())
                {
                        if (!depth_first(flags, vcall, src))
                        {
                                return false;
                        }
                }
                return true;
        }

        template <typename tindex>
        template <typename tvcall>
        bool digraph_t<tindex>::breadth_first(const tvcall& vcall) const
        {
                flags_t flags(vertices(), flag::none);
                for (const auto src : sources())
                {
                        if (!breadth_first(flags, vcall, src))
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
                // todo: have the input vertices consecutive!!!
                indices_t vindices(vertices(), 0);
                size_t index = vertices();
                depth_first([&] (const tindex vindex) { assert(index > 0); vindices[-- index] = vindex; });

                if (index != 0)
                {
                        vindices.clear();       // !DAG
                }
                return vindices;
        }

        template <typename tindex>
        bool digraph_t<tindex>::connected(const tindex src, const tindex dst) const
        {
                assert(src < vertices() && dst < vertices());

                if (src == dst)
                {
                        return std::find(m_edges.begin(), m_edges.end(), edge_t{src, dst}) != m_edges.end();
                }
                else
                {
                        flags_t flags(vertices(), flag::none);
                        return  depth_first(flags, [] (const tindex) {}, src) &&
                                flags[dst] == flag::visited;
                }
        }
}
