#pragma once

#include <deque>
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
                ///
                template <typename tvcall>
                void breadth_first(const tvcall&) const;

                ///
                /// \brief checks if the graph is a DAG
                ///
                bool dag() const;

                ///
                /// \brief returns the topologically sorted list of vertices
                ///
                indices_t tsort() const;

                ///
                /// \brief returns the topologically sorted list of vertices with
                ///     the inputs to any vertex being in the consecutive order
                ///
                indices_t tsort_strong() const;

                ///
                /// \brief check if there is a connected path from u to v
                ///
                bool connected(const size_t u, const size_t v) const;

                ///
                /// \brief check if the given vertices are sorted topologically
                ///
                bool tsort(const indices_t&) const;

                ///
                /// \brief check if the given vertices are sorted topologically and
                ///     the inputs to any vertex are in the consecutive order
                ///
                bool tsort_strong(const indices_t&) const;

                ///
                /// \brief returns the number of vertices
                ///
                size_t vertices() const { return m_vertices; }

        private:

                enum class color : uint8_t
                {
                        white,
                        gray,
                        black
                };

                struct info_t
                {
                        color   m_color;
                        size_t  m_depth;
                };
                using infos_t = std::vector<info_t>;

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
                bool depth_first(infos_t& infos, const tvcall& vcall, size_t u) const
                {
                        bool cyclic = false;
                        size_t depth = 0;

                        std::vector<size_t> q{u};
                        while (!q.empty())
                        {
                                u = q.back();
                                q.pop_back();

                                infos[u] = {color::black, depth ++};
                                vcall(u, infos[u].m_depth);

                                for (size_t v = 0; v < m_vertices; ++ v)
                                {
                                        if (get(u, v))
                                        {
                                                switch (infos[v].m_color)
                                                {
                                                case color::white:
                                                        infos[v].m_color = color::gray;
                                                        q.push_back(v);
                                                        break;
                                                default:
                                                        cyclic = true;
                                                        break;
                                                }
                                        }
                                }
                        }

                        return !cyclic;
                }

                template <typename tvcall>
                void breadth_first(infos_t& infos, const tvcall& vcall, size_t u) const
                {
                        infos[u] = {color::gray, 0};

                        std::deque<size_t> q{u};
                        while (!q.empty())
                        {
                                u = q.front();
                                q.pop_front();

                                for (size_t v = 0; v < m_vertices; ++ v)
                                {
                                        if (get(u, v) && infos[v].m_color == color::white)
                                        {
                                                infos[v].m_color = color::gray;
                                                infos[v].m_depth = infos[u].m_depth + 1;
                                                q.push_back(v);
                                        }
                                }

                                infos[u].m_color = color::black;
                                vcall(u, infos[u].m_depth);
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
                infos_t infos(m_vertices, {color::white, 0});

                bool cyclic = false;
                for (const auto u : sources())
                {
                        if (!depth_first(infos, vcall, u))
                        {
                                cyclic = true;
                        }
                }

                for (size_t u = 0; u < m_vertices; ++ u)
                {
                        if (infos[u].m_color == color::white && !depth_first(infos, vcall, u))
                        {
                                cyclic = true;
                        }
                }

                return !cyclic;
        }

        template <typename tvcall>
        void digraph_t::breadth_first(const tvcall& vcall) const
        {
                infos_t infos(m_vertices, {color::white, 0});
                for (const auto u : sources())
                {
                        breadth_first(infos, vcall, u);
                }

                for (size_t u = 0; u < m_vertices; ++ u)
                {
                        if (infos[u].m_color == color::white)
                        {
                                breadth_first(infos, vcall, u);
                        }
                }
        }

        inline bool digraph_t::dag() const
        {
                return !depth_first([] (const size_t, const size_t) {});
        }

        inline digraph_t::indices_t digraph_t::tsort() const
        {
                indices_t indices;
                if (dag())
                {
                        indices.resize(m_vertices, 0);
                        size_t index = 0;
                        depth_first([&] (const size_t u, const size_t)
                        {
                                assert(index < m_vertices);
                                indices[index ++] = u;
                        });
                        assert(index == m_vertices);
                }
                return indices;
        }

        inline digraph_t::indices_t digraph_t::tsort_strong() const
        {
                // todo
                return tsort();
        }

        inline bool digraph_t::connected(const size_t u, const size_t v) const
        {
                assert(u < m_vertices && v < m_vertices);

                if (get(u, v))
                {
                        return true;
                }
                else if (u == v)
                {
                        return false;
                }
                else
                {
                        infos_t infos(m_vertices, {color::white, 0});
                        depth_first(infos, [] (const size_t, const size_t) {}, u);
                        return infos[v].m_color == color::black;
                }
        }

        inline bool digraph_t::tsort(const indices_t& tsort) const
        {
                for (size_t ivertex = 0; ivertex < tsort.size(); ++ ivertex)
                {
                        const auto vertex = tsort[ivertex];

                        const auto sources = in(vertex);
                        for (const auto source : sources)
                        {
                                const auto it = std::find(tsort.begin(), tsort.end(), source);
                                const auto isource = static_cast<size_t>(std::distance(tsort.begin(), it));

                                // the source vertices should be before the vertex in the topological order!
                                if (isource > ivertex)
                                {
                                        return false;
                                }
                        }
                }

                return tsort.size() == m_vertices;
        }

        inline bool digraph_t::tsort_strong(const indices_t& tsort) const
        {
                for (size_t ivertex = 0; ivertex < tsort.size(); ++ ivertex)
                {
                        const auto vertex = tsort[ivertex];

                        size_t min_isource = tsort.size();
                        size_t max_isource = 0;

                        const auto sources = in(vertex);
                        for (const auto source : sources)
                        {
                                const auto it = std::find(tsort.begin(), tsort.end(), source);
                                const auto isource = static_cast<size_t>(std::distance(tsort.begin(), it));
                                min_isource = std::min(min_isource, isource);
                                max_isource = std::max(max_isource, isource);

                                // the source vertices should be before the vertex in the topological order!
                                if (isource > ivertex)
                                {
                                        return false;
                                }
                        }

                        // the source vertices should be consecutive in the topological order!
                        if (!sources.empty() && max_isource - min_isource + 1 != sources.size())
                        {
                                return false;
                        }
                }

                return tsort.size() == m_vertices;
        }
}
