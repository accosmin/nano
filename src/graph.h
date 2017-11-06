#pragma once

#include <vector>
#include <ostream>
#include <algorithm>

namespace nano
{
        ///
        /// \brief a vertex in the graph.
        ///
        template <typename tpayload>
        struct vertex_t
        {
                vertex_t() = default;
                vertex_t(const int id, tpayload&& data) : m_id(id), m_data(std::move(data)) {}

                int             m_id{0};        ///<
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
        /// \brief a directed edge in the graph.
        ///
        struct edge_t
        {
                edge_t() = default;
                edge_t(const int src, const int dst) : m_src(src), m_dst(dst) {}

                int             m_src{0};       ///< source vertex id
                int             m_dst{0};       ///< destination vertex id
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
        class graph_t
        {
        public:
                using edges_t = std::vector<edge_t>;
                using vertices_t = std::vector<vertex_t<tpayload>>;

                ///
                /// \brief add a new vertex
                /// \return the id of the new vertex
                ///
                int add(tpayload vertex);

                ///
                /// \brief create a directed edge between the src and the dst vertex ids
                /// \return true if the vertex ids are valid
                ///
                bool connect(const int src, const int dst);

                ///
                /// \brief checks if the graph has cycles
                ///
                bool cyclic() const;

                ///
                /// \brief checks if the graph is a tree
                ///
                bool tree() const;

                ///
                /// \brief topologically sort the graph
                ///
                void sort();

                ///
                /// \brief access functions
                ///
                const auto& edges() const { return m_edges; }
                const auto& vertices() const { return m_vertices; }
                bool empty() const { return m_edges.empty() && m_vertices.empty(); }

        private:

                auto vertex(const int id) const
                {
                        return  std::find_if(m_vertices.begin(), m_vertices.end(),
                                [=] (const auto& vertex) { return vertex.m_id == id; });
                }

                auto edge(const int src, const int dst) const
                {
                        return  std::find_if(m_edges.begin(), m_edges.end(),
                                [=] (const auto& edge) { return edge.m_src == src && edge.m_dst == dst; });
                }

                // attributes
                int             m_id{0};
                edges_t         m_edges;
                vertices_t      m_vertices;
        };

        template <typename tpayload>
        int graph_t<tpayload>::add(tpayload vertex)
        {
                m_vertices.emplace_back(++ m_id, std::move(vertex));
                return m_id;
        }

        template <typename tpayload>
        bool graph_t<tpayload>::connect(const int src, const int dst)
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
        bool graph_t<tpayload>::cyclic() const
        {
                return true;
        }
}
