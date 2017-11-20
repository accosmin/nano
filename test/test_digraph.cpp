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

NANO_CASE(topological)
{
        {
                digraph_t g;

                NANO_CHECK(g.dag());
                NANO_CHECK_EQUAL(g.tsort(), make_indices());
        }
        {
                digraph_t g;
                g.edge(0, 2);
                g.edge(2, 3);
                g.edge(1, 2);
                g.edge(3, 4);
                g.done();

                NANO_CHECK(g.dag());
                NANO_CHECK_EQUAL(g.tsort(), make_indices(1u, 0u, 2u, 3u, 4u));
        }
        {
                digraph_t g;
                g.edge(0, 2);
                g.edge(2, 3);
                g.edge(1, 2);
                g.edge(3, 0);
                g.done();

                NANO_CHECK(!g.dag());
                NANO_CHECK_EQUAL(g.tsort(), make_indices());
        }
        {
                digraph_t g;
                g.edge(1, 2);
                g.edge(2, 0);
                g.edge(0, 1);
                g.done();

                NANO_CHECK(!g.dag());
                NANO_CHECK_EQUAL(g.tsort(), make_indices());
        }
        {
                digraph_t g;
                g.edge(1, 2);
                g.edge(2, 0);
                g.done();

                NANO_CHECK(g.dag());
                NANO_CHECK_EQUAL(g.tsort(), make_indices(1u, 2u, 0u));
        }
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
        }
        {
                digraph_t g;
                g.edge(0, 2);
                g.edge(1, 2);
                g.edge(2, 3);
                g.edge(4, 5);
                g.edge(5, 6);
                g.done();

                NANO_CHECK(g.dag());
                NANO_CHECK_EQUAL(g.tsort(), make_indices(4u, 5u, 6u, 1u, 0u, 2u, 3u));
        }
}

NANO_END_MODULE()
