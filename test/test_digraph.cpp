#include "utest.h"
#include "digraph.h"

using namespace nano;

NANO_BEGIN_MODULE(test_digraph)

NANO_CASE(vertices)
{
        digraph_t<int> g;
        NANO_CHECK(g.empty());
        NANO_CHECK(g.edges().empty());
        NANO_CHECK(g.vertices().empty());

        NANO_CHECK_EQUAL(g.add(11), 0);
        NANO_CHECK_EQUAL(g.add(12), 1);
        NANO_CHECK_EQUAL(g.add(13), 2);
        NANO_CHECK_EQUAL(g.add(14), 3);

        NANO_CHECK(!g.empty());
        NANO_REQUIRE(g.edges().empty());
        NANO_REQUIRE_EQUAL(g.vertices().size(), 4);

        const auto v1 = vertex_t<int>{0, 11}; NANO_CHECK_EQUAL(g.vertices()[0], v1);
        const auto v2 = vertex_t<int>{1, 12}; NANO_CHECK_EQUAL(g.vertices()[1], v2);
        const auto v3 = vertex_t<int>{2, 13}; NANO_CHECK_EQUAL(g.vertices()[2], v3);
        const auto v4 = vertex_t<int>{3, 14}; NANO_CHECK_EQUAL(g.vertices()[3], v4);
}

NANO_CASE(vertices_and_edges)
{
        digraph_t<int> g;
        NANO_CHECK(g.empty());
        NANO_CHECK(g.edges().empty());
        NANO_CHECK(g.vertices().empty());

        NANO_CHECK_EQUAL(g.add(21), 0);
        NANO_CHECK_EQUAL(g.add(22), 1);
        NANO_CHECK_EQUAL(g.add(23), 2);
        NANO_CHECK_EQUAL(g.add(24), 3);

        NANO_CHECK(g.connect(0, 2));
        NANO_CHECK(g.connect(2, 3));
        NANO_CHECK(!g.connect(4, 3));   ///< invalid id
        NANO_CHECK(g.connect(1, 2));
        NANO_CHECK(!g.connect(1, 2));   ///< duplicated edge

        NANO_CHECK(!g.empty());
        NANO_REQUIRE_EQUAL(g.edges().size(), 3);
        NANO_REQUIRE_EQUAL(g.vertices().size(), 4);

        const auto e1 = edge_t{0, 2}; NANO_REQUIRE_EQUAL(g.edges()[0], e1);
        const auto e2 = edge_t{2, 3}; NANO_REQUIRE_EQUAL(g.edges()[1], e2);
        const auto e3 = edge_t{1, 2}; NANO_REQUIRE_EQUAL(g.edges()[2], e3);

        const auto v1 = vertex_t<int>{0, 21}; NANO_CHECK_EQUAL(g.vertices()[0], v1);
        const auto v2 = vertex_t<int>{1, 22}; NANO_CHECK_EQUAL(g.vertices()[1], v2);
        const auto v3 = vertex_t<int>{2, 23}; NANO_CHECK_EQUAL(g.vertices()[2], v3);
        const auto v4 = vertex_t<int>{3, 24}; NANO_CHECK_EQUAL(g.vertices()[3], v4);
}

NANO_CASE(cyclic0)
{
        digraph_t<int> g;
        NANO_CHECK_EQUAL(g.add(21), 0);
        NANO_CHECK_EQUAL(g.add(22), 1);
        NANO_CHECK_EQUAL(g.add(23), 2);
        NANO_CHECK_EQUAL(g.add(24), 3);

        NANO_CHECK(!g.cyclic());
}

NANO_CASE(cyclic1)
{
        digraph_t<int> g;
        NANO_CHECK_EQUAL(g.add(21), 0);
        NANO_CHECK_EQUAL(g.add(22), 1);
        NANO_CHECK_EQUAL(g.add(23), 2);
        NANO_CHECK_EQUAL(g.add(24), 3);

        NANO_CHECK(g.connect(0, 2));
        NANO_CHECK(g.connect(2, 3));
        NANO_CHECK(g.connect(1, 2));

        NANO_CHECK(!g.cyclic());
}

NANO_CASE(cyclic2)
{
        digraph_t<int> g;
        NANO_CHECK_EQUAL(g.add(21), 0);
        NANO_CHECK_EQUAL(g.add(22), 1);
        NANO_CHECK_EQUAL(g.add(23), 2);
        NANO_CHECK_EQUAL(g.add(24), 3);

        NANO_CHECK(g.connect(0, 2));
        NANO_CHECK(g.connect(2, 3));
        NANO_CHECK(g.connect(1, 2));
        NANO_CHECK(g.connect(3, 0));

        NANO_CHECK(g.cyclic());
}

NANO_CASE(cyclic3)
{
        digraph_t<int> g;
        NANO_CHECK_EQUAL(g.add(21), 0);
        NANO_CHECK_EQUAL(g.add(22), 1);
        NANO_CHECK_EQUAL(g.add(23), 2);

        NANO_CHECK(g.connect(1, 2));
        NANO_CHECK(g.connect(2, 0));
        NANO_CHECK(g.connect(0, 1));

        NANO_CHECK(g.cyclic());
}

NANO_CASE(cyclic4)
{
        digraph_t<int> g;
        NANO_CHECK_EQUAL(g.add(21), 0);
        NANO_CHECK_EQUAL(g.add(22), 1);
        NANO_CHECK_EQUAL(g.add(23), 2);

        NANO_CHECK(g.connect(1, 2));
        NANO_CHECK(g.connect(2, 0));

        NANO_CHECK(!g.cyclic());
}

NANO_CASE(cyclic5)
{
        digraph_t<int> g;
        NANO_CHECK_EQUAL(g.add(21), 0);
        NANO_CHECK_EQUAL(g.add(22), 1);
        NANO_CHECK_EQUAL(g.add(23), 2);
        NANO_CHECK_EQUAL(g.add(24), 3);
        NANO_CHECK_EQUAL(g.add(25), 4);
        NANO_CHECK_EQUAL(g.add(26), 5);
        NANO_CHECK_EQUAL(g.add(27), 6);

        NANO_CHECK(g.connect(0, 2));
        NANO_CHECK(g.connect(1, 2));
        NANO_CHECK(g.connect(2, 3));
        NANO_CHECK(g.connect(4, 5));
        NANO_CHECK(g.connect(5, 6));
        NANO_CHECK(g.connect(6, 4));

        NANO_CHECK(g.cyclic());
}

NANO_CASE(cyclic6)
{
        digraph_t<int> g;
        NANO_CHECK_EQUAL(g.add(21), 0);
        NANO_CHECK_EQUAL(g.add(22), 1);
        NANO_CHECK_EQUAL(g.add(23), 2);
        NANO_CHECK_EQUAL(g.add(24), 3);
        NANO_CHECK_EQUAL(g.add(25), 4);
        NANO_CHECK_EQUAL(g.add(26), 5);
        NANO_CHECK_EQUAL(g.add(27), 6);

        NANO_CHECK(g.connect(0, 2));
        NANO_CHECK(g.connect(1, 2));
        NANO_CHECK(g.connect(2, 3));
        NANO_CHECK(g.connect(4, 5));
        NANO_CHECK(g.connect(5, 6));

        NANO_CHECK(!g.cyclic());
}

NANO_END_MODULE()
