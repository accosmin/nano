#include "utest.h"
#include "graph.h"

using namespace nano;

NANO_BEGIN_MODULE(test_graph)

NANO_CASE(vertices)
{
        graph_t<int> g;
        NANO_CHECK(g.empty());
        NANO_CHECK(g.edges().empty());
        NANO_CHECK(g.vertices().empty());

        NANO_CHECK_EQUAL(g.add(11), 1);
        NANO_CHECK_EQUAL(g.add(12), 2);
        NANO_CHECK_EQUAL(g.add(13), 3);
        NANO_CHECK_EQUAL(g.add(14), 4);

        NANO_CHECK(!g.empty());
        NANO_REQUIRE(g.edges().empty());
        NANO_REQUIRE_EQUAL(g.vertices().size(), 4);

        const auto v1 = vertex_t<int>{1, 11}; NANO_CHECK_EQUAL(g.vertices()[0], v1);
        const auto v2 = vertex_t<int>{2, 12}; NANO_CHECK_EQUAL(g.vertices()[1], v2);
        const auto v3 = vertex_t<int>{3, 13}; NANO_CHECK_EQUAL(g.vertices()[2], v3);
        const auto v4 = vertex_t<int>{4, 14}; NANO_CHECK_EQUAL(g.vertices()[3], v4);
}

NANO_CASE(vertices_and_edges)
{
        graph_t<int> g;
        NANO_CHECK(g.empty());
        NANO_CHECK(g.edges().empty());
        NANO_CHECK(g.vertices().empty());

        NANO_CHECK_EQUAL(g.add(21), 1);
        NANO_CHECK_EQUAL(g.add(22), 2);
        NANO_CHECK_EQUAL(g.add(23), 3);
        NANO_CHECK_EQUAL(g.add(24), 4);

        NANO_CHECK(g.connect(1, 3));
        NANO_CHECK(g.connect(3, 4));
        NANO_CHECK(!g.connect(5, 4));   ///< invalid id
        NANO_CHECK(g.connect(2, 3));
        NANO_CHECK(!g.connect(2, 3));   ///< duplicated edge

        NANO_CHECK(!g.empty());
        NANO_REQUIRE_EQUAL(g.edges().size(), 3);
        NANO_REQUIRE_EQUAL(g.vertices().size(), 4);

        const auto e1 = edge_t{1, 3}; NANO_REQUIRE_EQUAL(g.edges()[0], e1);
        const auto e2 = edge_t{3, 4}; NANO_REQUIRE_EQUAL(g.edges()[1], e2);
        const auto e3 = edge_t{2, 3}; NANO_REQUIRE_EQUAL(g.edges()[2], e3);

        const auto v1 = vertex_t<int>{1, 21}; NANO_CHECK_EQUAL(g.vertices()[0], v1);
        const auto v2 = vertex_t<int>{2, 22}; NANO_CHECK_EQUAL(g.vertices()[1], v2);
        const auto v3 = vertex_t<int>{3, 23}; NANO_CHECK_EQUAL(g.vertices()[2], v3);
        const auto v4 = vertex_t<int>{4, 24}; NANO_CHECK_EQUAL(g.vertices()[3], v4);
}

NANO_CASE(cyclic)
{
        graph_t<int> g;
        NANO_CHECK_EQUAL(g.add(21), 1);
        NANO_CHECK_EQUAL(g.add(22), 2);
        NANO_CHECK_EQUAL(g.add(23), 3);
        NANO_CHECK_EQUAL(g.add(24), 4);

        NANO_CHECK(g.connect(1, 3));
        NANO_CHECK(g.connect(3, 4));
        NANO_CHECK(g.connect(2, 3));

        NANO_CHECK(g.cyclic());
}

NANO_END_MODULE()
