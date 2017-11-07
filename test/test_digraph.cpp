#include "utest.h"
#include "digraph.h"

using namespace nano;

std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& indices)
{
        for (const auto index : indices)
        {
                os << index << ',';
        }
        return os;
}

NANO_BEGIN_MODULE(test_digraph)

NANO_CASE(vertices)
{
        digraph_t<int> g;
        NANO_CHECK(g.empty());
        NANO_CHECK(g.edges().empty());
        NANO_CHECK(g.vertices().empty());

        g.vertices(10, 11, 12, 13);

        NANO_CHECK(!g.empty());
        NANO_REQUIRE(g.edges().empty());
        NANO_REQUIRE_EQUAL(g.vertices().size(), 4);

        const auto v1 = vertex_t<int>{10}; NANO_CHECK_EQUAL(g.vertices()[0], v1);
        const auto v2 = vertex_t<int>{11}; NANO_CHECK_EQUAL(g.vertices()[1], v2);
        const auto v3 = vertex_t<int>{12}; NANO_CHECK_EQUAL(g.vertices()[2], v3);
        const auto v4 = vertex_t<int>{13}; NANO_CHECK_EQUAL(g.vertices()[3], v4);
}

NANO_CASE(vertices_and_edges)
{
        digraph_t<int> g;
        NANO_CHECK(g.empty());
        NANO_CHECK(g.edges().empty());
        NANO_CHECK(g.vertices().empty());

        g.vertices(20, 21, 22, 23);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(!g.edge(4, 3));   ///< invalid id
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(!g.edge(1, 2));   ///< duplicated edge

        NANO_CHECK(!g.empty());
        NANO_REQUIRE_EQUAL(g.edges().size(), 3);
        NANO_REQUIRE_EQUAL(g.vertices().size(), 4);

        const auto e1 = edge_t{0, 2}; NANO_REQUIRE_EQUAL(g.edges()[0], e1);
        const auto e2 = edge_t{2, 3}; NANO_REQUIRE_EQUAL(g.edges()[1], e2);
        const auto e3 = edge_t{1, 2}; NANO_REQUIRE_EQUAL(g.edges()[2], e3);

        const auto v1 = vertex_t<int>{20}; NANO_CHECK_EQUAL(g.vertices()[0], v1);
        const auto v2 = vertex_t<int>{21}; NANO_CHECK_EQUAL(g.vertices()[1], v2);
        const auto v3 = vertex_t<int>{22}; NANO_CHECK_EQUAL(g.vertices()[2], v3);
        const auto v4 = vertex_t<int>{23}; NANO_CHECK_EQUAL(g.vertices()[3], v4);
}

NANO_CASE(incoming)
{
        digraph_t<int> g;
        g.vertices(20, 21, 22, 23, 24);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(3, 4));

        const std::vector<size_t> incoming = {0, 1};
        NANO_CHECK_EQUAL(g.incoming(), incoming);
}

NANO_CASE(outgoing)
{
        digraph_t<int> g;
        g.vertices(20, 21, 22, 23, 24);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(3, 4));

        const std::vector<size_t> outgoing = {4};
        NANO_CHECK_EQUAL(g.outgoing(), outgoing);
}

NANO_CASE(cyclic0)
{
        digraph_t<int> g;
        NANO_CHECK(g.dag());

        g.vertices(20, 21, 22, 23);
        NANO_CHECK(g.dag());
}

NANO_CASE(cyclic1)
{
        digraph_t<int> g;
        g.vertices(20, 21, 22, 23, 24);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(3, 4));

        NANO_CHECK(g.dag());
}

NANO_CASE(cyclic2)
{
        digraph_t<int> g;
        g.vertices(20, 21, 22, 23);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(3, 0));

        NANO_CHECK(!g.dag());
}

NANO_CASE(cyclic3)
{
        digraph_t<int> g;
        g.vertices(20, 21, 22);

        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(2, 0));
        NANO_CHECK(g.edge(0, 1));

        NANO_CHECK(!g.dag());
}

NANO_CASE(cyclic4)
{
        digraph_t<int> g;
        g.vertices(20, 21, 22);

        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(2, 0));

        NANO_CHECK(g.dag());
}

NANO_CASE(cyclic5)
{
        digraph_t<int> g;
        g.vertices(20, 21, 22, 23, 24, 25, 26);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(g.edge(4, 5));
        NANO_CHECK(g.edge(5, 6));
        NANO_CHECK(g.edge(6, 4));

        NANO_CHECK(!g.dag());
}

NANO_CASE(cyclic6)
{
        digraph_t<int> g;
        g.vertices(20, 21, 22, 23, 24, 25, 26);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(g.edge(4, 5));
        NANO_CHECK(g.edge(5, 6));

        NANO_CHECK(g.dag());
}

NANO_END_MODULE()
