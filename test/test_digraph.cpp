#include "utest.h"
#include "digraph.h"

using namespace nano;

template <typename tindex>
std::ostream& operator<<(std::ostream& os, const std::vector<tindex>& indices)
{
        for (const auto index : indices)
        {
                os << index << ',';
        }
        return os;
}

NANO_BEGIN_MODULE(test_digraph)

NANO_CASE(empty)
{
        digraph_t<size_t> g;

        NANO_CHECK(g.empty());
        NANO_CHECK(g.edges().empty());
        NANO_CHECK_EQUAL(g.vertices(), 0u);
}

NANO_CASE(vertices)
{
        digraph_t<size_t> g(4);

        NANO_CHECK(!g.empty());
        NANO_REQUIRE(g.edges().empty());
        NANO_REQUIRE_EQUAL(g.vertices(), 4u);
}

NANO_CASE(vertices_and_edges)
{
        digraph_t<size_t> g(4);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(!g.edge(4, 3));   ///< invalid id
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(!g.edge(1, 2));   ///< duplicated edge

        NANO_CHECK(!g.empty());
        NANO_REQUIRE_EQUAL(g.edges().size(), 3u);
        NANO_REQUIRE_EQUAL(g.vertices(), 4u);

        const auto e1 = edge_t<size_t>{0, 2}; NANO_REQUIRE_EQUAL(g.edges()[0], e1);
        const auto e2 = edge_t<size_t>{2, 3}; NANO_REQUIRE_EQUAL(g.edges()[1], e2);
        const auto e3 = edge_t<size_t>{1, 2}; NANO_REQUIRE_EQUAL(g.edges()[2], e3);
}

NANO_CASE(incoming)
{
        digraph_t<size_t> g(5);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(3, 4));

        const std::vector<size_t> incoming = {0, 1};
        NANO_CHECK_EQUAL(g.incoming(), incoming);
}

NANO_CASE(outgoing)
{
        digraph_t<size_t> g(5);

        NANO_CHECK(g.edge(0, 2));
        NANO_CHECK(g.edge(2, 3));
        NANO_CHECK(g.edge(1, 2));
        NANO_CHECK(g.edge(3, 4));

        const std::vector<size_t> outgoing = {4};
        NANO_CHECK_EQUAL(g.outgoing(), outgoing);
}

NANO_CASE(dag)
{
        {
                digraph_t<size_t> g;

                NANO_CHECK(g.dag());
        }
        {
                digraph_t<size_t> g(4);

                NANO_CHECK(g.dag());
        }
        {
                digraph_t<size_t> g(5);

                NANO_CHECK(g.edge(0, 2));
                NANO_CHECK(g.edge(2, 3));
                NANO_CHECK(g.edge(1, 2));
                NANO_CHECK(g.edge(3, 4));

                NANO_CHECK(g.dag());
        }
        {
                digraph_t<size_t> g(4);

                NANO_CHECK(g.edge(0, 2));
                NANO_CHECK(g.edge(2, 3));
                NANO_CHECK(g.edge(1, 2));
                NANO_CHECK(g.edge(3, 0));

                NANO_CHECK(!g.dag());
        }
        {
                digraph_t<size_t> g(3);

                NANO_CHECK(g.edge(1, 2));
                NANO_CHECK(g.edge(2, 0));
                NANO_CHECK(g.edge(0, 1));

                NANO_CHECK(!g.dag());
        }
        {
                digraph_t<size_t> g(3);

                NANO_CHECK(g.edge(1, 2));
                NANO_CHECK(g.edge(2, 0));

                NANO_CHECK(g.dag());
        }
        {
                digraph_t<size_t> g(7);

                NANO_CHECK(g.edge(0, 2));
                NANO_CHECK(g.edge(1, 2));
                NANO_CHECK(g.edge(2, 3));
                NANO_CHECK(g.edge(4, 5));
                NANO_CHECK(g.edge(5, 6));
                NANO_CHECK(g.edge(6, 4));

                NANO_CHECK(!g.dag());
        }
        {
                digraph_t<size_t> g(7);

                NANO_CHECK(g.edge(0, 2));
                NANO_CHECK(g.edge(1, 2));
                NANO_CHECK(g.edge(2, 3));
                NANO_CHECK(g.edge(4, 5));
                NANO_CHECK(g.edge(5, 6));

                NANO_CHECK(g.dag());
        }
}

NANO_END_MODULE()
