#include "utest.h"
#include "digraph.h"

using index_t = size_t;
using digraph_t = nano::digraph_t<index_t>;
using edge_t = digraph_t::edge_t;
using indices_t = digraph_t::indices_t;

std::ostream& operator<<(std::ostream& os, const indices_t& indices)
{
        for (const auto index : indices)
        {
                os << index << ',';
        }
        return os;
}

std::ostream& operator<<(std::ostream& os, const edge_t& e)
{
        return os << '{' << e.first << "->" << e.second << '}';
}

edge_t make_edge(const index_t src, const index_t dst)
{
        return {src, dst};
}

indices_t make_indices()
{
        return {};
}

template <typename... tindices>
indices_t make_indices(const tindices... indices)
{
        return {indices...};
}

indices_t depth_first(const digraph_t& graph)
{
        indices_t vertices;
        graph.depth_first([&] (const index_t vertex) { vertices.push_back(vertex); });
        return vertices;
}

indices_t breadth_first(const digraph_t& graph)
{
        indices_t vertices;
        graph.breadth_first([&] (const index_t vertex) { vertices.push_back(vertex); });
        return vertices;
}

void check_tsort(const digraph_t& graph)
{
        const auto tsort = graph.tsort();
        for (size_t ivertex = 0; ivertex < tsort.size(); ++ ivertex)
        {
                const auto vertex = tsort[ivertex];

                size_t min_isource = tsort.size();
                size_t max_isource = 0;

                const auto sources = graph.in(vertex);
                for (const auto source : sources)
                {
                        const auto it = std::find(tsort.begin(), tsort.end(), source);
                        const auto isource = static_cast<size_t>(std::distance(tsort.begin(), it));
                        min_isource = std::min(min_isource, isource);
                        max_isource = std::max(max_isource, isource);

                        // the source vertices should be before the vertex in the topological order!
                        NANO_CHECK_LESS_EQUAL(isource, ivertex);
                }

                // the source vertices should be consecutive in the topological order!
                if (!sources.empty())
                {
                        std::cout << "vertex = " << vertex << ", [" << min_isource << ", " << max_isource << "]" << std::endl;
                        NANO_CHECK_EQUAL(max_isource - min_isource + 1, sources.size());
                }
        }
}

NANO_BEGIN_MODULE(test_digraph)

NANO_CASE(empty)
{
        digraph_t g;

        NANO_CHECK(g.empty());
        NANO_CHECK(g.edges().empty());
        NANO_CHECK_EQUAL(g.vertices(), 0u);
}

NANO_CASE(vertices)
{
        digraph_t g;
        g.edge(0, 2);
        g.edge(2, 3);
        g.edge(1, 2);
        g.edge(1, 2);   ///< duplicated edge
        g.done();

        NANO_CHECK(!g.empty());
        NANO_REQUIRE_EQUAL(g.edges().size(), 3u);
        NANO_REQUIRE_EQUAL(g.vertices(), 4u);

        const auto e1 = make_edge(0u, 2u); NANO_CHECK_EQUAL(g.edges()[0], e1);
        const auto e2 = make_edge(1u, 2u); NANO_CHECK_EQUAL(g.edges()[1], e2);
        const auto e3 = make_edge(2u, 3u); NANO_CHECK_EQUAL(g.edges()[2], e3);
}

NANO_CASE(in)
{
        digraph_t g;
        g.edge(0, 2);
        g.edge(2, 3);
        g.edge(1, 2);
        g.edge(3, 4);
        g.done();

        NANO_CHECK_EQUAL(g.sources(), make_indices(0u, 1u));
        NANO_CHECK_EQUAL(g.in(0), make_indices());
        NANO_CHECK_EQUAL(g.in(1), make_indices());
        NANO_CHECK_EQUAL(g.in(2), make_indices(0u, 1u));
        NANO_CHECK_EQUAL(g.in(3), make_indices(2u));
        NANO_CHECK_EQUAL(g.in(4), make_indices(3u));
}

NANO_CASE(out)
{
        digraph_t g;
        g.edge(0, 2);
        g.edge(2, 3);
        g.edge(1, 2);
        g.edge(3, 4);
        g.done();

        NANO_CHECK_EQUAL(g.sinks(), make_indices(4u));
        NANO_CHECK_EQUAL(g.out(0), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(1), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(2), make_indices(3u));
        NANO_CHECK_EQUAL(g.out(3), make_indices(4u));
        NANO_CHECK_EQUAL(g.out(4), make_indices());
}

NANO_CASE(topo1)
{
        digraph_t g;

        NANO_CHECK(g.dag());
        NANO_CHECK_EQUAL(g.tsort(), make_indices());
        check_tsort(g);
        NANO_CHECK_EQUAL(depth_first(g), make_indices());
        NANO_CHECK_EQUAL(breadth_first(g), make_indices());
}

NANO_CASE(topo2)
{
        digraph_t g;
        g.edge(0, 2);
        g.edge(2, 3);
        g.edge(1, 2);
        g.edge(3, 4);
        g.done();

        NANO_CHECK(g.dag());
        NANO_CHECK(!g.connected(0, 1));
        NANO_CHECK(!g.connected(1, 0));
        NANO_CHECK(g.connected(0, 2));
        NANO_CHECK(!g.connected(2, 0));
        NANO_CHECK(g.connected(0, 3));
        NANO_CHECK(g.connected(0, 4));
        NANO_CHECK(g.connected(1, 2));
        NANO_CHECK(g.connected(1, 3));
        NANO_CHECK(g.connected(1, 4));
        NANO_CHECK(g.connected(2, 3));
        NANO_CHECK(g.connected(2, 4));
        NANO_CHECK(g.connected(3, 4));
        NANO_CHECK_EQUAL(g.tsort(), make_indices(1u, 0u, 2u, 3u, 4u));
        check_tsort(g);
        NANO_CHECK_EQUAL(depth_first(g), make_indices(0u, 2u, 3u, 4u, 1u));
        NANO_CHECK_EQUAL(breadth_first(g), make_indices(0u, 2u, 3u, 4u, 1u));
}

/*NANO_CASE(topo3)
{
        digraph_t g;
        g.edge(0, 2);
        g.edge(2, 3);
        g.edge(1, 2);
        g.edge(3, 0);
        g.done();

        NANO_CHECK(!g.dag());
        NANO_CHECK_EQUAL(g.tsort(), make_indices());
        check_tsort(g);
}

NANO_CASE(topo4)
{
        digraph_t g;
        g.edge(1, 2);
        g.edge(2, 0);
        g.edge(0, 1);
        g.done();

        NANO_CHECK(!g.dag());
        NANO_CHECK_EQUAL(g.tsort(), make_indices());
        check_tsort(g);
}

NANO_CASE(topo5)
{
        digraph_t g;
        g.edge(1, 2);
        g.edge(2, 0);
        g.done();

        NANO_CHECK(g.dag());
        NANO_CHECK_EQUAL(g.tsort(), make_indices(1u, 2u, 0u));
        check_tsort(g);
}

NANO_CASE(topo6)
{
        digraph_t g;
        g.edge(0, 2);
        g.edge(1, 2);
        g.edge(2, 3);
        g.edge(4, 5);
        g.edge(5, 6);
        g.edge(6, 4);
        g.done();

        NANO_CHECK(!g.dag());
        NANO_CHECK_EQUAL(g.tsort(), make_indices());
        check_tsort(g);
}

NANO_CASE(topo7)
{
        digraph_t g;
        g.edge(0, 2);
        g.edge(1, 2);
        g.edge(2, 3);
        g.edge(4, 5);
        g.edge(5, 6);
        g.done();

        NANO_CHECK(g.dag());
        NANO_CHECK(!g.connected(0, 4));
        NANO_CHECK(!g.connected(0, 5));
        NANO_CHECK(!g.connected(0, 6));
        NANO_CHECK(!g.connected(2, 4));
        NANO_CHECK(!g.connected(2, 5));
        NANO_CHECK(!g.connected(2, 6));
        NANO_CHECK(g.connected(4, 5));
        NANO_CHECK(!g.connected(5, 4));
        NANO_CHECK(g.connected(4, 6));
        NANO_CHECK(!g.connected(6, 4));
        NANO_CHECK(g.connected(5, 6));
        NANO_CHECK(!g.connected(6, 5));
        NANO_CHECK(!g.connected(6, 6));
        NANO_CHECK(!g.connected(4, 2));
        NANO_CHECK_EQUAL(g.tsort(), make_indices(4u, 5u, 6u, 1u, 0u, 2u, 3u));
        check_tsort(g);
}

NANO_CASE(topo8)
{
        // see: https://en.wikipedia.org/wiki/Topological_sorting#/media/File:Directed_acyclic_graph_2.svg
        digraph_t g;
        g.edge(0, 3);
        g.edge(1, 3);
        g.edge(1, 4);
        g.edge(2, 4);
        g.edge(3, 5);
        g.edge(3, 6);
        g.edge(3, 7);
        g.edge(2, 7);
        g.edge(4, 6);
        g.done();

        NANO_CHECK(g.dag());
        NANO_CHECK_EQUAL(g.tsort(), make_indices(2u, 1u, 4u, 0u, 3u, 7u, 6u, 5u));
        check_tsort(g);
}*/

NANO_CASE(topo9)
{
        digraph_t g;
        g.edge(0, 1);
        g.edge(0, 2);
        g.edge(0, 5);
        g.edge(1, 3);
        g.edge(2, 4);
        g.edge(3, 6);
        g.edge(4, 6);
        g.edge(5, 6);
        g.done();

        NANO_CHECK(g.dag());
        NANO_CHECK(!g.connected(0, 1));
        NANO_CHECK(!g.connected(1, 0));
        NANO_CHECK(g.connected(0, 2));
        NANO_CHECK(!g.connected(2, 0));
        NANO_CHECK(g.connected(0, 3));
        NANO_CHECK(g.connected(0, 4));
        NANO_CHECK(g.connected(1, 2));
        NANO_CHECK(g.connected(1, 3));
        NANO_CHECK(g.connected(1, 4));
        NANO_CHECK(g.connected(2, 3));
        NANO_CHECK(g.connected(2, 4));
        NANO_CHECK(g.connected(3, 4));
        check_tsort(g);
        NANO_CHECK_EQUAL(depth_first(g), make_indices(0u, 1u, 3u, 6u, 2u, 3u, 5u));
        NANO_CHECK_EQUAL(breadth_first(g), make_indices(0u, 1u, 2u, 5u, 3u, 4u, 6u));
}

NANO_END_MODULE()
