#pragma once

#include <deque>
#include <stack>
#include <vector>
#include <cassert>
#include <cstdint>

namespace nano
{
        ///
        /// \brief generic directed graph specified by a set of edges.
        ///     todo: faster (aka lower complexity) for ::sink & ::sinks
        ///
        class digraph_t
        {
        public:
                using indices_t = std::vector<size_t>;

                ///
                /// \brief constructor
                ///
                explicit digraph_t(const size_t vertices);

                ///
                /// \brief disable copying
                ///
                digraph_t(const digraph_t&) = delete;
                digraph_t& operator=(const digraph_t&) = delete;

                ///
                /// \brief enable moving
                ///
                digraph_t(digraph_t&&) = default;
                digraph_t& operator=(digraph_t&&) = default;

                ///
                /// \brief create a directed edge (u, v)
                ///
                void edge(const size_t u, const size_t v);

                ///
                /// \brief create the directed edges to connect the given chain of vertex ids
                ///
                template <typename... tindices>
                void edge(const size_t u, const size_t v, const tindices... rest)
                {
                        edge(u, v);
                        edge(v, rest...);
                }

                ///
                /// \brief returns the source vertices (aka the vertices without incoming vertices)
                ///
                indices_t sources() const;

                ///
                /// \brief returns the incoming vertices of the given vertex id
                ///
                indices_t in(const size_t v) const;

                ///
                /// \brief returns the sink vertices (aka the vertices without outgoing vertices)
                ///
                indices_t sinks() const;

                ///
                /// \brief returns the outgoing vertices of the given vertex id
                ///
                indices_t out(const size_t u) const;

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
                /// \brief check if there is a connected path from u to v
                ///
                bool connected(const size_t u, const size_t v) const;

        private:

                enum class flag : uint8_t
                {
                        white,
                        gray,
                        black
                };

                using flags_t = std::vector<flag>;

                bool get(const size_t u, const size_t v) const
                {
                        assert(u < m_vertices && v < m_vertices);
                        return m_adjmat[u * m_vertices + v];
                }

                void set(const size_t u, const size_t v, const bool connected)
                {
                        assert(u < m_vertices && v < m_vertices);
                        m_adjmat[u * m_vertices + v] = connected;
                }

                template <typename tvcall>
                bool depth_first(flags_t& flags, const tvcall& vcall, size_t u) const
                {
                        std::stack<size_t> q;
                        q.push(u);

                        while (!q.empty())
                        {
                                u = q.top();
                                q.pop();

                                flags[u] = flag::black;
                                vcall(u);

                                for (size_t v = 0; v < m_vertices; ++ v)
                                {
                                        if (get(u, v))
                                        {
                                                if (flags[v] == flag::white)
                                                {
                                                        flags[v] = flag::gray;
                                                        q.push(v);
                                                }
                                                else if (flags[v] == flag::black)
                                                {
                                                        return false;
                                                }
                                        }
                                }
                        }

                        return true;
                }

                template <typename tvcall>
                void breadth_first(flags_t& flags, const tvcall& vcall, size_t u) const
                {
                        flags[u] = flag::gray;

                        std::deque<size_t> q{u};
                        while (!q.empty())
                        {
                                u = q.front();
                                q.pop_front();

                                for (size_t v = 0; v < m_vertices; ++ v)
                                {
                                        if (get(u, v) && flags[v] == flag::white)
                                        {
                                                flags[v] = flag::gray;
                                                q.push_back(v);
                                        }
                                }

                                flags[u] = flag::black;
                                vcall(u);
                        }
                }

                // attributes
                size_t                  m_vertices;     ///< number of vertices
                std::vector<bool>       m_adjmat;       ///< adjacency matrix
        };

        inline digraph_t::digraph_t(const size_t vertices) :
                m_vertices(vertices),
                m_adjmat(vertices * vertices, false)
        {
        }

        inline void digraph_t::edge(const size_t u, const size_t v)
        {
                set(u, v, true);
        }

        inline digraph_t::indices_t digraph_t::sources() const
        {
                indices_t srcs;
                for (size_t v = 0; v < m_vertices; ++ v)
                {
                        bool source = true;
                        for (size_t u = 0; u < m_vertices && source; ++ u)
                        {
                                source = !get(u, v);
                        }
                        if (source)
                        {
                                srcs.push_back(v);
                        }
                }

                return srcs;
        }

        inline digraph_t::indices_t digraph_t::in(const size_t v) const
        {
                indices_t srcs;
                for (size_t u = 0; u < m_vertices; ++ u)
                {
                        if (get(u, v))
                        {
                                srcs.push_back(u);
                        }
                }

                return srcs;
        }

        inline digraph_t::indices_t digraph_t::sinks() const
        {
                indices_t dsts;
                for (size_t u = 0; u < m_vertices; ++ u)
                {
                        bool sink = true;
                        for (size_t v = 0; v < m_vertices && sink; ++ v)
                        {
                                sink = !get(u, v);
                        }
                        if (sink)
                        {
                                dsts.push_back(u);
                        }
                }

                return dsts;
        }

        inline digraph_t::indices_t digraph_t::out(const size_t u) const
        {
                indices_t dsts;
                for (size_t v = 0; v < m_vertices; ++ v)
                {
                        if (get(u, v))
                        {
                                dsts.push_back(v);
                        }
                }

                return dsts;
        }

        template <typename tvcall>
        bool digraph_t::depth_first(const tvcall& vcall) const
        {
                flags_t flags(m_vertices, flag::white);
                for (const auto u : sources())
                {
                        if (!depth_first(flags, vcall, u))
                        {
                                return false;
                        }
                }
                return true;
        }

        template <typename tvcall>
        bool digraph_t::breadth_first(const tvcall& vcall) const
        {
                flags_t flags(m_vertices, flag::white);
                for (const auto u : sources())
                {
                        breadth_first(flags, vcall, u);
                }
                return true;
        }

        inline bool digraph_t::dag() const
        {
                return !depth_first([] (const size_t) {});
        }

        inline digraph_t::indices_t digraph_t::tsort() const
        {
                // todo: have the input vertices consecutive!!!
                return indices_t{};
                /*
                indices_t vindices(vertices(), 0);
                size_t index = vertices();
                depth_first([&] (const size_t vindex) { assert(index > 0); vindices[-- index] = vindex; });

                if (index != 0)
                {
                        vindices.clear();       // !DAG
                }
                return vindices;*/
        }

        inline bool digraph_t::connected(const size_t u, const size_t v) const
        {
                assert(u < m_vertices && v < m_vertices);

                if (get(u, v))
                {
                        return true;
                }
                else
                {
                        flags_t flags(m_vertices, flag::white);
                        depth_first(flags, [] (const size_t) {}, u);
                        return flags[v] == flag::black;
                }
        }
}
