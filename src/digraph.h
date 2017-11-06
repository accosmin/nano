#pragma once

#include <vector>
#include <ostream>
#include <algorithm>

namespace nano
{
        ///
        /// \brief a vertex in a directed graph.
        ///
        template <typename tpayload>
        struct vertex_t
        {
                vertex_t() = default;
                vertex_t(const size_t id, tpayload&& data) : m_id(id), m_data(std::move(data)) {}

                size_t          m_id{0};        ///<
                tpayload        m_data;         ///<
        };

        template <typename tpayload>
        bool operator==(const vertex_t<tpayload>& v1, const vertex_t<tpayload>& v2)
        {
                return v1.m_id == v2.m_id && v1.m_data == v2.m_data;
        }

        template <typename tpayload>
        std::ostream& operator<<(std::ostream& os, const vertex_t<tpayload>& v)
        {
                return os << '{' << v.m_id << ':' << v.m_data << '}';
        }

        ///
        /// \brief a directed edge in a directed graph.
        ///
        struct edge_t
        {
                edge_t() = default;
                edge_t(const size_t src, const size_t dst) : m_src(src), m_dst(dst) {}

                size_t          m_src{0};       ///< source vertex id
                size_t          m_dst{0};       ///< destination vertex id
        };

        inline bool operator==(const edge_t& e1, const edge_t& e2)
        {
                return e1.m_src == e2.m_src && e1.m_dst == e2.m_dst;
        }

        std::ostream& operator<<(std::ostream& os, const edge_t& e)
        {
                return os << '{' << e.m_src << "->" << e.m_dst << '}';
        }

        ///
        /// \brief generic directed graph where each nodes has a given payload.
        ///
        template <typename tpayload>
        class digraph_t
        {
        public:
                using edges_t = std::vector<edge_t>;
                using vertices_t = std::vector<vertex_t<tpayload>>;

                ///
                /// \brief add a new vertex
                /// \return the id of the new vertex
                ///
                size_t add(tpayload vertex);

                ///
                /// \brief add new vertices
                /// \return the id of the last vertex in the pack
                ///
                template <typename... tpayloads>
                size_t add(tpayload vertex, tpayloads&&... vertices)
                {
                        add(vertex);
                        return add(vertices...);
                }

                ///
                /// \brief create a directed edge between the src and the dst vertex ids
                /// \return true if the vertex ids are valid
                ///
                bool connect(const size_t src, const size_t dst);

                ///
                /// \brief returns then vertices with no incoming edge
                ///
                std::vector<size_t> sources() const;

                ///
                /// \brief returns the vertices with no outgoing edge
                ///
                std::vector<size_t> destinations() const;

                ///
                /// \brief depth-first search where the given operator is called with the current vertex id
                /// \return true if the graph is not a DAG
                ///
                template <typename tvcall>
                bool depth_first(const tvcall&) const;

                ///
                /// \brief checks if the graph has cycles (aka not a DAG)
                ///
                bool cyclic() const { return !dag(); }

                ///
                /// \brief checks if the graph is a DAG
                ///
                bool dag() const;

                ///
                /// \brief topologically sort the graph
                /// \return true if sorting makes sense (e.g. DAG)
                ///
                bool tsort();

                ///
                /// \brief access functions
                ///
                const auto& edges() const { return m_edges; }
                const auto& vertices() const { return m_vertices; }
                bool empty() const { return m_edges.empty() && m_vertices.empty(); }

        private:

                auto vertex(const size_t id) const
                {
                        return  std::find_if(m_vertices.begin(), m_vertices.end(),
                                [=] (const auto& vertex) { return vertex.m_id == id; });
                }

                auto edge(const size_t src, const size_t dst) const
                {
                        return  std::find_if(m_edges.begin(), m_edges.end(),
                                [=] (const auto& edge) { return edge.m_src == src && edge.m_dst == dst; });
                }

                enum class flag : uint8_t
                {
                        none,
                        temporary,
                        permanent,
                };

                template <typename tvcall>
                bool visit(std::vector<flag>& flags, const tvcall& vcall, const size_t vindex) const
                {
                        assert(flags.size() == m_vertices.size());

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
                                        if (edge.m_src == vindex && !visit(flags, vcall, edge.m_dst))
                                        {
                                                return false;
                                        }
                                }
                                flags[vindex] = flag::permanent;
                                vcall(vindex);
                                return true;
                        }
                }

                static std::vector<size_t> difference(std::vector<size_t>& set1, std::vector<size_t>& set2)
                {
                        std::sort(set1.begin(), set1.end());
                        std::sort(set2.begin(), set2.end());

                        set1.erase(std::unique(set1.begin(), set1.end()), set1.end());
                        set2.erase(std::unique(set2.begin(), set2.end()), set2.end());

                        std::vector<size_t> diff;
                        std::set_difference(
                                set1.begin(), set1.end(), set2.begin(), set2.end(),
                                std::inserter(diff, diff.begin()));
                        return diff;
                }

                // attributes
                size_t          m_id{0};
                edges_t         m_edges;
                vertices_t      m_vertices;
        };

        template <typename tpayload>
        size_t digraph_t<tpayload>::add(tpayload vertex)
        {
                m_vertices.emplace_back(m_id, std::move(vertex));
                return m_id ++;
        }

        template <typename tpayload>
        bool digraph_t<tpayload>::connect(const size_t src, const size_t dst)
        {
                if (    vertex(src) != m_vertices.end() &&
                        vertex(dst) != m_vertices.end() &&
                        edge(src, dst) == m_edges.end())
                {
                        m_edges.emplace_back(src, dst);
                        return true;
                }
                else
                {
                        return false;
                }
        }

        template <typename tpayload>
        std::vector<size_t> digraph_t<tpayload>::sources() const
        {
                std::vector<size_t> srcs, dsts;
                srcs.reserve(m_edges.size());
                dsts.reserve(m_edges.size());
                for (const auto& edge : m_edges)
                {
                        srcs.emplace_back(edge.m_src);
                        dsts.emplace_back(edge.m_dst);
                }

                return difference(srcs, dsts);
        }

        template <typename tpayload>
        std::vector<size_t> digraph_t<tpayload>::destinations() const
        {
                std::vector<size_t> srcs, dsts;
                srcs.reserve(m_edges.size());
                dsts.reserve(m_edges.size());
                for (const auto& edge : m_edges)
                {
                        srcs.emplace_back(edge.m_src);
                        dsts.emplace_back(edge.m_dst);
                }

                return difference(dsts, srcs);
        }

        template <typename tpayload>
        template <typename tvcall>
        bool digraph_t<tpayload>::depth_first(const tvcall& vcall) const
        {
                std::vector<flag> flags(m_vertices.size(), flag::none);

                // start from the vertices that don't have incoming edges
                for (const auto src : sources())
                {
                        if (!visit(flags, vcall, src))
                        {
                                return false;
                        }
                }
                return true;
        }

        template <typename tpayload>
        bool digraph_t<tpayload>::dag() const
        {
                return depth_first([] (const size_t vindex) { (void)vindex; });
        }

        template <typename tpayload>
        bool digraph_t<tpayload>::tsort()
        {
                std::vector<size_t> vindices;
                depth_first([&] (const size_t vindex) { vindices.push_back(vindex); });

                if (vindices.size() == m_vertices.size())
                {
                        return true;
                }
                else
                {
                        return false;
                }
        }
}
