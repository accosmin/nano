#include "utest.h"
#include "digraph.h"

using namespace nano;

using color = digraph_t::color;
using cycle = digraph_t::cycle;
using infos_t = digraph_t::infos_t;
using indices_t = digraph_t::indices_t;
using conn_t = std::vector<std::pair<size_t, size_t>>;

std::ostream& operator<<(std::ostream& os, const indices_t& indices)
{
        for (const auto index : indices)
        {
                os << index << ',';
        }
        return os;
}

std::ostream& operator<<(std::ostream& os, const infos_t& infos)
{
        os << std::endl;
        for (const auto& info : infos)
        {
                switch (info.m_color)
                {
                case color::white:      os << "w:"; break;
                case color::black:      os << "b:"; break;
                }
                switch (info.m_cycle)
                {
                case cycle::none:       os << "n:"; break;
                case cycle::detected:   os << "d:"; break;
                }
                os << static_cast<uint32_t>(info.m_comp) << ':' << static_cast<uint32_t>(info.m_depth) << ',';
        }
        return os << std::endl;
}

template <typename... tindices>
indices_t make_indices(const tindices... indices)
{
        return {indices...};
}

template <typename... tinfos>
infos_t make_infos(const tinfos&... infos)
{
        return {infos...};
}

void check_conn(const digraph_t& g, const conn_t& conn)
{
        for (size_t u = 0; u < g.vertices(); ++ u)
        {
                for (size_t v = 0; v < g.vertices(); ++ v)
                {
                        const bool connected = std::find(conn.begin(), conn.end(), std::make_pair(u, v)) != conn.end();
                        NANO_CHECK_EQUAL(g.connected(u, v), connected);
                }
        }
}

NANO_BEGIN_MODULE(test_digraph)

NANO_CASE(graph0)
{
        digraph_t g(3);
        g.edge(0u, 0u, 1u, 2u); // cycle here
        NANO_CHECK_EQUAL(g.vertices(), 3u);

        const conn_t conn =
        {
                {0u, 0u}, {0u, 1u}, {0u, 2u},
                {1u, 2u}
        };
        check_conn(g, conn);

        NANO_CHECK_EQUAL(g.sources(), make_indices());
        NANO_CHECK_EQUAL(g.in(0), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(1), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(2), make_indices(1u));

        NANO_CHECK_EQUAL(g.sinks(), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(0), make_indices(0u, 1u));
        NANO_CHECK_EQUAL(g.out(1), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(2), make_indices());

        const infos_t infos =
        {
                {color::black, cycle::detected, 0u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 2u, 0u}
        };
        NANO_CHECK_EQUAL(g.visit(), infos);

        NANO_CHECK(!g.dag());
}

NANO_CASE(graph1)
{
        digraph_t g(3);
        g.edge(0u, 1u, 2u, 0u); // cycle here
        NANO_CHECK_EQUAL(g.vertices(), 3u);

        const conn_t conn =
        {
                {0u, 1u}, {0u, 2u},
                {1u, 2u}, {1u, 0u},
                {2u, 0u}, {2u, 1u}
        };
        check_conn(g, conn);

        NANO_CHECK_EQUAL(g.sources(), make_indices());
        NANO_CHECK_EQUAL(g.in(0), make_indices(2u));
        NANO_CHECK_EQUAL(g.in(1), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(2), make_indices(1u));

        NANO_CHECK_EQUAL(g.sinks(), make_indices());
        NANO_CHECK_EQUAL(g.out(0), make_indices(1u));
        NANO_CHECK_EQUAL(g.out(1), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(2), make_indices(0u));

        const infos_t infos =
        {
                {color::black, cycle::detected, 0u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 2u, 0u}
        };
        NANO_CHECK_EQUAL(g.visit(), infos);

        NANO_CHECK(!g.dag());
}

NANO_CASE(graph2)
{
        digraph_t g(5);
        g.edge(0u, 1u, 2u);
        g.edge(3u, 2u);
        g.edge(3u, 4u);
        NANO_CHECK_EQUAL(g.vertices(), 5u);

        const conn_t conn =
        {
                {0u, 1u}, {0u, 2u},
                {1u, 2u},
                {3u, 2u}, {3u, 4u}
        };
        check_conn(g, conn);

        NANO_CHECK_EQUAL(g.sources(), make_indices(0u, 3u));
        NANO_CHECK_EQUAL(g.in(0), make_indices());
        NANO_CHECK_EQUAL(g.in(1), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(2), make_indices(1u, 3u));
        NANO_CHECK_EQUAL(g.in(3), make_indices());
        NANO_CHECK_EQUAL(g.in(4), make_indices(3u));

        NANO_CHECK_EQUAL(g.sinks(), make_indices(2u, 4u));
        NANO_CHECK_EQUAL(g.out(0), make_indices(1u));
        NANO_CHECK_EQUAL(g.out(1), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(2), make_indices());
        NANO_CHECK_EQUAL(g.out(3), make_indices(2u, 4u));
        NANO_CHECK_EQUAL(g.out(4), make_indices());

        const infos_t infos =
        {
                {color::black, cycle::none, 0u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 2u, 0u},
                {color::black, cycle::none, 0u, 0u},
                {color::black, cycle::none, 1u, 0u},
        };
        NANO_CHECK_EQUAL(g.visit(), infos);

        NANO_CHECK(g.dag());
        NANO_CHECK(g.tsort(g.tsort()));

        NANO_CHECK(g.tsort(make_indices(0u, 3u, 4u, 1u, 2u)));
        NANO_CHECK(g.tsort(make_indices(0u, 1u, 3u, 2u, 4u)));
        NANO_CHECK(g.tsort(make_indices(0u, 3u, 1u, 2u, 4u)));
        NANO_CHECK(g.tsort(make_indices(0u, 3u, 1u, 4u, 2u)));
}

NANO_CASE(graph3)
{
        digraph_t g(6);
        g.edge(0u, 1u);
        g.edge(3u, 1u, 2u);
        g.edge(3u, 4u, 5u, 2u);
        NANO_CHECK_EQUAL(g.vertices(), 6u);

        const conn_t conn =
        {
                {0u, 1u}, {0u, 2u},
                {1u, 2u},
                {3u, 1u}, {3u, 2u}, {3u, 4u}, {3u, 5u},
                {4u, 5u}, {4u, 2u},
                {5u, 2u}
        };
        check_conn(g, conn);

        NANO_CHECK_EQUAL(g.sources(), make_indices(0u, 3u));
        NANO_CHECK_EQUAL(g.in(0), make_indices());
        NANO_CHECK_EQUAL(g.in(1), make_indices(0u, 3u));
        NANO_CHECK_EQUAL(g.in(2), make_indices(1u, 5u));
        NANO_CHECK_EQUAL(g.in(3), make_indices());
        NANO_CHECK_EQUAL(g.in(4), make_indices(3u));
        NANO_CHECK_EQUAL(g.in(5), make_indices(4u));

        NANO_CHECK_EQUAL(g.sinks(), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(0), make_indices(1u));
        NANO_CHECK_EQUAL(g.out(1), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(2), make_indices());
        NANO_CHECK_EQUAL(g.out(3), make_indices(1u, 4u));
        NANO_CHECK_EQUAL(g.out(4), make_indices(5u));
        NANO_CHECK_EQUAL(g.out(5), make_indices(2u));

        const infos_t infos =
        {
                {color::black, cycle::none, 0u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 3u, 0u},
                {color::black, cycle::none, 0u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 2u, 0u}
        };
        NANO_CHECK_EQUAL(g.visit(), infos);

        NANO_CHECK(g.dag());
        NANO_CHECK(g.tsort(g.tsort()));
}

NANO_CASE(graph4)
{
        digraph_t g(5);
        g.edge(3u, 4u, 0u, 1u, 2u, 4u); // cycle here
        NANO_CHECK_EQUAL(g.vertices(), 5u);

        const conn_t conn =
        {
                {0u, 1u}, {0u, 2u}, {0u, 4u},
                {1u, 2u}, {1u, 4u}, {1u, 0u},
                {2u, 4u}, {2u, 0u}, {2u, 1u},
                {3u, 4u}, {3u, 0u}, {3u, 1u}, {3u, 2u},
                {4u, 0u}, {4u, 1u}, {4u, 2u}
        };
        check_conn(g, conn);

        NANO_CHECK_EQUAL(g.sources(), make_indices(3u));
        NANO_CHECK_EQUAL(g.in(0), make_indices(4u));
        NANO_CHECK_EQUAL(g.in(1), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(2), make_indices(1u));
        NANO_CHECK_EQUAL(g.in(3), make_indices());
        NANO_CHECK_EQUAL(g.in(4), make_indices(2u, 3u));

        NANO_CHECK_EQUAL(g.sinks(), make_indices());
        NANO_CHECK_EQUAL(g.out(0), make_indices(1u));
        NANO_CHECK_EQUAL(g.out(1), make_indices(2u));
        NANO_CHECK_EQUAL(g.out(2), make_indices(4u));
        NANO_CHECK_EQUAL(g.out(3), make_indices(4u));
        NANO_CHECK_EQUAL(g.out(4), make_indices(0u));

        const infos_t infos =
        {
                {color::black, cycle::none, 2u, 0u},
                {color::black, cycle::none, 3u, 0u},
                {color::black, cycle::none, 4u, 0u},
                {color::black, cycle::none, 0u, 0u},
                {color::black, cycle::detected, 1u, 0u}
        };
        NANO_CHECK_EQUAL(g.visit(), infos);

        NANO_CHECK(!g.dag());
}

NANO_CASE(graph5)
{
        digraph_t g(7);
        g.edge(0u, 1u, 3u);
        g.edge(2u, 1u, 3u);
        g.edge(4u, 5u, 6u, 4u); // cycle here
        NANO_CHECK_EQUAL(g.vertices(), 7u);

        const conn_t conn =
        {
                {0u, 1u}, {0u, 3u},
                {1u, 3u},
                {2u, 1u}, {2u, 3u},
                {4u, 5u}, {4u, 6u},
                {5u, 6u}, {5u, 4u},
                {6u, 4u}, {6u, 5u}
        };
        check_conn(g, conn);

        NANO_CHECK_EQUAL(g.sources(), make_indices(0u, 2u));
        NANO_CHECK_EQUAL(g.in(0), make_indices());
        NANO_CHECK_EQUAL(g.in(1), make_indices(0u, 2u));
        NANO_CHECK_EQUAL(g.in(2), make_indices());
        NANO_CHECK_EQUAL(g.in(3), make_indices(1u));
        NANO_CHECK_EQUAL(g.in(4), make_indices(6u));
        NANO_CHECK_EQUAL(g.in(5), make_indices(4u));
        NANO_CHECK_EQUAL(g.in(6), make_indices(5u));

        NANO_CHECK_EQUAL(g.sinks(), make_indices(3u));
        NANO_CHECK_EQUAL(g.out(0), make_indices(1u));
        NANO_CHECK_EQUAL(g.out(1), make_indices(3u));
        NANO_CHECK_EQUAL(g.out(2), make_indices(1u));
        NANO_CHECK_EQUAL(g.out(3), make_indices());
        NANO_CHECK_EQUAL(g.out(4), make_indices(5u));
        NANO_CHECK_EQUAL(g.out(5), make_indices(6u));
        NANO_CHECK_EQUAL(g.out(6), make_indices(4u));

        const infos_t infos =
        {
                {color::black, cycle::none, 0u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 0u, 0u},
                {color::black, cycle::none, 2u, 0u},
                {color::black, cycle::detected, 0u, 1u},
                {color::black, cycle::none, 1u, 1u},
                {color::black, cycle::none, 2u, 1u}
        };
        NANO_CHECK_EQUAL(g.visit(), infos);

        NANO_CHECK(!g.dag());
}

NANO_CASE(graph6)
{
        digraph_t g(7);
        g.edge(0u, 1u, 3u, 6u);
        g.edge(0u, 2u, 4u, 6u);
        g.edge(0u, 5u, 6u);
        NANO_CHECK_EQUAL(g.vertices(), 7u);

        const conn_t conn =
        {
                {0u, 1u}, {0u, 2u}, {0u, 3u}, {0u, 4u}, {0u, 5u}, {0u, 6u},
                {1u, 3u}, {1u, 6u},
                {2u, 4u}, {2u, 6u},
                {3u, 6u},
                {4u, 6u},
                {5u, 6u}
        };
        check_conn(g, conn);

        NANO_CHECK_EQUAL(g.sources(), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(0), make_indices());
        NANO_CHECK_EQUAL(g.in(1), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(2), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(3), make_indices(1u));
        NANO_CHECK_EQUAL(g.in(4), make_indices(2u));
        NANO_CHECK_EQUAL(g.in(5), make_indices(0u));
        NANO_CHECK_EQUAL(g.in(6), make_indices(3u, 4u, 5u));

        NANO_CHECK_EQUAL(g.sinks(), make_indices(6u));
        NANO_CHECK_EQUAL(g.out(0), make_indices(1u, 2u, 5u));
        NANO_CHECK_EQUAL(g.out(1), make_indices(3u));
        NANO_CHECK_EQUAL(g.out(2), make_indices(4u));
        NANO_CHECK_EQUAL(g.out(3), make_indices(6u));
        NANO_CHECK_EQUAL(g.out(4), make_indices(6u));
        NANO_CHECK_EQUAL(g.out(5), make_indices(6u));
        NANO_CHECK_EQUAL(g.out(6), make_indices());

        const infos_t infos =
        {
                {color::black, cycle::none, 0u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 2u, 0u},
                {color::black, cycle::none, 2u, 0u},
                {color::black, cycle::none, 1u, 0u},
                {color::black, cycle::none, 3u, 0u}
        };
        NANO_CHECK_EQUAL(g.visit(), infos);

        NANO_CHECK(g.dag());
        NANO_CHECK(g.tsort(g.tsort()));
}

NANO_END_MODULE()
