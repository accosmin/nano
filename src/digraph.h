#pragma once

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

                enum class color : uint16_t
                {
                        white,
                        gray,
                        black
                };

                enum class cycle : uint16_t
                {
                        none,
                        detected
                };

                ///
                /// \brief information gathered per vertex when visiting the graph.
                ///
                struct info_t
                {
                        color   m_color{color::white};  ///<
                        cycle   m_cycle{cycle::none};   ///<
                        size_t  m_depth{0};             ///< depth (starting from the deepest source)
                        size_t  m_comp{0};              ///< component index
                };

                using infos_t = std::vector<info_t>;
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
                /// \brief visit all vertices of the graph
                ///
                infos_t visit() const;

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
                /// \brief returns the number of vertices
                ///
                size_t vertices() const { return m_vertices; }

        private:

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

                template <typename toperator>
                size_t foreach_in(const size_t v, const toperator& op) const
                {
                        size_t count = 0;
                        for (size_t u = 0; u < m_vertices; ++ u)
                        {
                                if (get(u, v))
                                {
                                        op(u);
                                        ++ count;
                                }
                        }
                        return count;
                }

                template <typename toperator>
                size_t foreach_out(const size_t u, const toperator& op) const
                {
                        size_t count = 0;
                        for (size_t v = 0; v < m_vertices; ++ v)
                        {
                                if (get(u, v))
                                {
                                        op(v);
                                        ++ count;
                                }
                        }
                        return count;
                }

                void visit(infos_t& infos, size_t u) const
                {
                        //todo: adjust the component index if connecting to another component: take its component index & -- comp
                        //todo: adjust the depth if connecting to a vertex having an already set depth

                        indices_t q{u};
                        while (!q.empty())
                        {
                                u = q.back();
                                q.pop_back();

                                infos[u].m_color = color::gray;
                                foreach_out(u, [&] (const size_t v)
                                {
                                        switch (infos[v].m_color)
                                        {
                                        case color::white:
                                                infos[v].m_depth = std::max(infos[v].m_depth, infos[u].m_depth + 1);
                                                infos[v].m_color = color::gray;
                                                q.push_back(v);
                                                break;
                                        case color::gray:
                                                infos[v].m_cycle = cycle::detected;
                                                break;
                                        case color::black:
                                                break;
                                        }
                                });
                                infos[u].m_color = color::black;
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
                        if (!foreach_in(v, [&] (const size_t) {}))
                        {
                                srcs.push_back(v);
                        }
                }

                return srcs;
        }

        inline digraph_t::indices_t digraph_t::in(const size_t v) const
        {
                indices_t srcs;
                foreach_in(v, [&] (const size_t u)
                {
                        srcs.push_back(u);

                });

                return srcs;
        }

        inline digraph_t::indices_t digraph_t::sinks() const
        {
                indices_t dsts;
                for (size_t u = 0; u < m_vertices; ++ u)
                {
                        if (!foreach_out(u, [&] (const size_t) {}))
                        {
                                dsts.push_back(u);
                        }
                }

                return dsts;
        }

        inline digraph_t::indices_t digraph_t::out(const size_t u) const
        {
                indices_t dsts;
                foreach_out(u, [&] (const size_t v)
                {
                        dsts.push_back(v);
                });

                return dsts;
        }

        digraph_t::infos_t digraph_t::visit() const
        {
                infos_t infos(m_vertices);
                for (const auto u : sources())
                {
                        visit(infos, u);
                }

                for (size_t u = 0; u < m_vertices; ++ u)
                {
                        if (infos[u].m_color == color::white)
                        {
                                visit(infos, u);
                        }
                }
                return infos;
        }

        inline bool digraph_t::dag() const
        {
                const auto infos = visit();
                const auto tester = [] (const info_t& info) { return info.m_cycle == cycle::detected; };
                return std::find_if(infos.begin(), infos.end(), tester) == infos.end();
        }

        inline digraph_t::indices_t digraph_t::tsort() const
        {
                // todo
                return indices_t{};
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
                        infos_t infos(m_vertices);
                        visit(infos, u);
                        return infos[v].m_color != color::white;
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

        inline bool operator==(const digraph_t::info_t& i1, const digraph_t::info_t& i2)
        {
                return  i1.m_color == i2.m_color &&
                        i1.m_depth == i2.m_depth &&
                        i1.m_comp == i2.m_comp;
        }
}
