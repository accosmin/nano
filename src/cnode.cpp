#include "cnode.h"
#include "logger.h"
#include "digraph.h"

using namespace nano;

template <typename tvalue>
static void reorder(std::vector<tvalue>& values, const indices_t& order)
{
        assert(values.size() == order.size());

        for (size_t i = 0; i < order.size(); ++ i)
        {
                size_t orig = order[i];
                while (i < orig)
                {
                        orig = order[orig];
                }
                if (i != orig)
                {
                        std::swap(values[i], values[orig]);
                }
        }
}

static void reindex(indices_t& indices, const indices_t& order)
{
        for (auto& index : indices)
        {
                index = order[index];
        }
}

cnode_t::cnode_t(const cnode_t& other) :
        m_name(other.m_name),
        m_type(other.m_type),
        m_node(other.m_node->clone()),
        m_inodes(other.m_inodes),
        m_onodes(other.m_onodes)
{
}

cnode_t::cnode_t(string_t name, string_t type, rlayer_t&& node) :
        m_name(std::move(name)),
        m_type(std::move(type)),
        m_node(std::move(node))
{
}

bool nano::check_nodes(cnodes_t& nodes)
{
        log_info() << "model: checking the computation graph...";

        // create the computation graph
        digraph_t graph(nodes.size());
        for (size_t src = 0; src < nodes.size(); ++ src)
        {
                for (const size_t dst : nodes[src].m_onodes)
                {
                        graph.edge(src, dst);
                }
        }

        if (nodes.empty())
        {
                log_error() << "model: expecting at least a node!";
                return false;
        }

        // and check it
        const auto sources = graph.sources();
        const auto sinks = graph.sinks();
        if (sinks.size() != 1)
        {
                // todo: may relax this condition (many outputs may be needed to solve reinforcement learning problems)
                log_error() << "model: expecting exactly one output node!";
                return false;
        }

        if (!graph.dag())
        {
                // todo: may relax this condition
                log_error() << "model: cyclic computation graphs are not supported!";
                return false;
        }

        for (const auto sink : sinks)
        {
                const auto it = std::find_if(sources.begin(), sources.end(), [&] (const size_t source)
                {
                        return graph.connected(source, sink);
                });

                if (it == sources.end())
                {
                        log_error() << "model: detected unreachable output node [" << nodes[sink].m_name << "]!";
                        return false;
                }
        }

        for (const auto source : sources)
        {
                const auto it = std::find_if(sinks.begin(), sinks.end(), [&] (const size_t sink)
                {
                        return graph.connected(source, sink);
                });

                if (it == sinks.end())
                {
                        log_error() << "model: detected unused input node [" << nodes[source].m_name << "]!";
                        return false;
                }
        }

        // OK, reorder nodes using the topological sorting
        const auto tsort = graph.tsort();
        // todo: check that the inputs to a node are consecutive (to pass idata as a block)
        reorder(nodes, tsort);
        for (auto& cnode : nodes)
        {
                // also reindex the inputs/outputs
                reindex(cnode.m_inodes, tsort);
                reindex(cnode.m_onodes, tsort);
        }
        return true;
}
