#include "model.h"
#include "logger.h"
#include <iostream>
#include "digraph.h"
#include "text/table.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include "math/numeric.h"
#include "tensor/numeric.h"
#include "text/algorithm.h"

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

rmodel_t model_t::clone() const
{
        return std::make_unique<model_t>(*this);
}

void model_t::clear()
{
        m_nodes.clear();
}

tensor_size_t model_t::xsize(const tensor_size_t count) const
{
        assert(!m_nodes.empty());

        tensor_size_t sum = nano::size(m_idims);
        for (const auto& cnode : m_nodes)
        {
                sum += nano::osize(cnode.m_node);
        }
        return sum * count;
}

void model_t::allocate(const tensor_size_t count)
{
        m_xdata.resize(xsize(count));

        tensor_size_t obegin = count * nano::size(m_idims);
        for (auto& cnode : m_nodes)
        {
                cnode.m_obegin = obegin;
                obegin += count * nano::osize(cnode.m_node);
        }

        assert(obegin == m_xdata.size());
}

size_t model_t::find_node(const string_t& name) const
{
        const auto it = std::find_if(m_nodes.begin(), m_nodes.end(), [&] (const auto& node) { return name == node.m_name; });
        return (it == m_nodes.end()) ? string_t::npos : static_cast<size_t>(std::distance(m_nodes.begin(), it));
}

bool model_t::add(const json_t& json)
{
        try
        {
                const auto name = json.at("name").get<string_t>();
                const auto type = json.at("type").get<string_t>();

                log_info() << "model: adding node [" << name << "] of type [" << type << "]...";
                if (find_node(name) != string_t::npos)
                {
                        log_error() << "model: duplicated node name [" << name << "]!";
                        return false;
                }

                auto node = get_layers().get(type);
                if (!node)
                {
                        log_error() << "model: invalid node type [" << type << "]!";
                        return false;
                }
                node->from_json(json);

                m_nodes.emplace_back(name, type, std::move(node));
                return true;
        }
        catch (std::exception& e)
        {
                log_error() << "model: failed to configure node [" << e.what() << "]!";
                return false;
        }
}

bool model_t::connect(const string_t& name1, const string_t& name2)
{
        log_info() << "model: connecting nodes [" << name1 << "] -> [" << name2 << "]...";

        const auto src = find_node(name1);
        if (src == string_t::npos)
        {
                log_error() << "model: unknown node [" << name1 << "]!";
                return false;
        }

        const auto dst = find_node(name2);
        if (dst == string_t::npos)
        {
                log_error() << "model: unknown node [" << name2 << "]!";
                return false;
        }

        m_nodes[dst].m_inodes.push_back(src);
        return true;
}

bool model_t::connect(const strings_t& names)
{
        for (size_t i = 0; i + 1 < names.size(); ++ i)
        {
                if (!connect(names[i], names[i + 1]))
                {
                        return false;
                }
        }
        return true;
}

bool model_t::from_json(const json_t& json)
{
        log_info() << "model: configuring...";

        clear();

        try
        {
                const auto& json_nodes = json.at("nodes");
                const auto& json_model = json.at("model");

                for (const auto& json_node : json_nodes)
                {
                        if (!add(json_node))
                        {
                                return false;
                        }
                }

                for (const auto& json_conn : json_model)
                {
                        for (size_t i = 0; i + 1 < json_conn.size(); ++ i)
                        {
                                if (!connect(json_conn[i].get<string_t>(), json_conn[i + 1].get<string_t>()))
                                {
                                        return false;
                                }
                        }
                }
        }
        catch (std::exception& e)
        {
                log_error() << "model: failed to configure node [" << e.what() << "]!";
                return false;
        }

        return done();
}

bool model_t::done()
{
        log_info() << "model: checking the computation graph...";

        // create the computation graph
        digraph_t graph(m_nodes.size());
        for (size_t dst = 0; dst < m_nodes.size(); ++ dst)
        {
                for (const size_t src : m_nodes[dst].m_inodes)
                {
                        graph.edge(src, dst);
                }
        }

        if (m_nodes.empty())
        {
                log_error() << "model: expecting at least a node!";
                clear();
                return false;
        }

        // and check it
        const auto sources = graph.sources();
        const auto sinks = graph.sinks();
        if (sinks.size() != 1)
        {
                // todo: may relax this condition (many outputs may be needed to solve reinforcement learning problems)
                log_error() << "model: expecting exactly one output node!";
                clear();
                return false;
        }

        if (!graph.dag())
        {
                // todo: may relax this condition
                log_error() << "model: cyclic computation graphs are not supported!";
                clear();
                return false;
        }

        for (const auto sink : sinks)
        {
                const auto it = std::find_if(sources.begin(), sources.end(), [&] (const size_t source)
                {
                        return graph.connected(source, sink);
                });

                if (it == sources.end() && graph.vertices() > 1)
                {
                        log_error() << "model: detected unreachable output node [" << m_nodes[sink].m_name << "]!";
                        clear();
                        return false;
                }
        }

        for (const auto source : sources)
        {
                const auto it = std::find_if(sinks.begin(), sinks.end(), [&] (const size_t sink)
                {
                        return graph.connected(source, sink);
                });

                if (it == sinks.end() && graph.vertices() > 1)
                {
                        log_error() << "model: detected unused input node [" << m_nodes[source].m_name << "]!";
                        clear();
                        return false;
                }
        }

        // OK, reorder nodes using the topological sorting
        const auto tsort = graph.tsort();
        reorder(m_nodes, tsort);
        for (auto& cnode : m_nodes)
        {
                // also reindex their inputs
                reindex(cnode.m_inodes, tsort);
        }
        return true;
}

json_t model_t::to_json() const
{
        json_t json;

        auto&& json_nodes = json["nodes"];
        for (size_t i = 0; i < m_nodes.size(); ++ i)
        {
                const auto& name = m_nodes[i].m_name;
                const auto& type = m_nodes[i].m_type;
                const auto& node = m_nodes[i].m_node;

                auto&& json_node = json_nodes[name];
                json_node["type"] = type;
                node->to_json(json_node);
        }

        auto&& json_model = json["model"];
        std::vector<std::pair<size_t, size_t>> edges;
        for (size_t dst = 0; dst < m_nodes.size(); ++ dst)
        {
                for (const auto src : m_nodes[dst].m_inodes)
                {
                        edges.emplace_back(src, dst);
                }
        }
        for (size_t i = 0; i < edges.size(); ++ i)
        {
                const auto src = edges[i].first;
                const auto dst = edges[i].second;

                const auto& name1 = m_nodes[src].m_name;
                const auto& name2 = m_nodes[dst].m_name;
                json_model[i] = {name1, name2};
        }

        return json;
}

bool model_t::save(const string_t& path) const
{
        const auto json = to_json();

        obstream_t ob(path);
        return  ob.write(idims()) &&
                ob.write(odims()) &&
                ob.write(json.dump()) &&
                ob.write_vector(params());
}

bool model_t::load(const string_t& path)
{
        tensor3d_dim_t idims, odims;
        vector_t pdata;
        string_t json;

        ibstream_t ib(path);
        return  ib.read(idims) &&
                ib.read(odims) &&
                ib.read(json) &&
                ib.read_vector(pdata) &&
                from_json(json_t::parse(json)) &&
                resize(idims, odims) &&
                pdata.size() == psize() &&
                [&] () { params(pdata); return true; }();
}

void model_t::params(const vector_t& pdata)
{
        assert(pdata.size() == psize());
        m_pdata = pdata;
        m_gdata.setZero();
}

void model_t::random()
{
        for (const auto& cnode : m_nodes)
        {
                if (cnode.m_node->psize() > 0)
                {
                        cnode.m_node->random(cnode.pdata(m_pdata));
                }
        }
        m_gdata.setZero();

        assert(m_pdata.array().isFinite().all());
}

tensor4d_cmap_t model_t::output(const tensor4d_t& idata)
{
        assert(idata.tensor(0).dims() == idims());
        assert(m_pdata.array().isFinite().all());
        assert(idata.array().isFinite().all());
        assert(!m_nodes.empty());

        const auto count = idata.size<0>();
        m_probe_output.measure([&] ()
        {
                // allocate buffers if the count (aka the number of samples to process at once) changed
                if (m_xdata.size() != xsize(count))
                {
                        allocate(count);
                }

                // forward step
                cnode_t::idata(m_xdata, count, m_idims) = idata;
                for (auto& cnode : m_nodes)
                {
                        cnode.output(
                                cnode.idata(cxdata(), count, m_nodes, m_idims),
                                cnode.pdata(cpdata()),
                                cnode.odata(m_xdata, count));
                }
        }, count);

        assert(m_xdata.array().isFinite().all());
        assert(m_pdata.array().isFinite().all());

        return onode().odata(cxdata(), count);
}

const vector_t& model_t::gparam(const tensor4d_t& odata)
{
        assert(odata.array().isFinite().all());
        assert(m_xdata.array().isFinite().all());
        assert(m_pdata.array().isFinite().all());
        assert(odata.tensor(0).dims() == odims());
        assert(!m_nodes.empty());

        const auto count = odata.size<0>();
        m_probe_gparam.measure([&] ()
        {
                assert(m_xdata.size() == xsize(count));

                // backward step
                onode().odata(m_xdata, count) = odata;
                for (auto it = m_nodes.rbegin(); it != m_nodes.rend(); ++ it)
                {
                        auto& cnode = *it;
                        cnode.gparam(
                                cnode.idata(cxdata(), count, m_nodes, m_idims),
                                cnode.pdata(m_gdata),
                                cnode.odata(cxdata(), count));
                        if (!cnode.m_inodes.empty())
                        {
                                cnode.ginput(
                                        cnode.idata(m_xdata, count, m_nodes, m_idims),
                                        cnode.pdata(cpdata()),
                                        cnode.odata(cxdata(), count));
                        }
                }
        }, count);

        assert(m_xdata.array().isFinite().all());
        assert(m_pdata.array().isFinite().all());
        assert(m_gdata.array().isFinite().all());

        return m_gdata;
}

bool model_t::resize(const tensor3d_dim_t& idims, const tensor3d_dim_t& odims)
{
        log_info() << "model: resizing the computation nodes...";

        // resize computation nodes starting from the input
        for (auto& cnode : m_nodes)
        {
                std::vector<tensor3d_dim_t> cidims;
                if (cnode.m_inodes.empty())
                {
                        cidims.push_back(idims);
                }
                else
                {
                        for (const auto inode : cnode.m_inodes)
                        {
                                cidims.push_back(m_nodes[inode].m_node->odims());
                        }
                }

                if (!cnode.m_node->resize(cidims))
                {
                        log_error() << "model: failed to resize node [" << cnode.m_name << "]!";
                        return false;
                }

                cnode.m_probe_output = probe_t{cnode.m_name, cnode.m_name + "(output)", cnode.m_node->flops_output()};
                cnode.m_probe_ginput = probe_t{cnode.m_name, cnode.m_name + "(ginput)", cnode.m_node->flops_ginput()};
                cnode.m_probe_gparam = probe_t{cnode.m_name, cnode.m_name + "(gparam)", cnode.m_node->flops_gparam()};
        }

        // check output size to match the target
        if (m_nodes.rbegin()->m_node->odims() != odims)
        {
                log_error() << "model: miss-matching output size "
                        << m_nodes.rbegin()->m_node->odims() << ", expecting " << odims << "!";
                return false;
        }

        m_idims = idims;
        m_odims = odims;

        // allocate buffers & setup probes
        tensor_size_t psize = 0;
        int64_t flops_output = 0;
        int64_t flops_ginput = 0;
        int64_t flops_gparam = 0;

        for (const auto& cnode : m_nodes)
        {
                psize += cnode.m_node->psize();
                flops_output += cnode.m_node->flops_output();
                flops_ginput += cnode.m_node->flops_ginput();
                flops_gparam += cnode.m_node->flops_gparam();
        }

        m_pdata.resize(psize);
        m_gdata.resize(psize);
        m_probe_output = probe_t{"model", "model(output)", flops_output};
        m_probe_ginput = probe_t{"model", "model(ginput)", flops_ginput};
        m_probe_gparam = probe_t{"model", "model(gparam)", flops_gparam};

        tensor_size_t pbegin = 0;
        for (auto& cnode : m_nodes)
        {
                cnode.m_pbegin = pbegin;
                pbegin += cnode.m_node->psize();
        }
        assert(pbegin == m_pdata.size());

        return true;
}

strings_t model_t::node_names(const indices_t& indices) const
{
        strings_t names;
        names.reserve(indices.size());
        for (const auto index : indices)
        {
                names.push_back(m_nodes[index].m_name);
        }

        return names;
}

void model_t::describe() const
{
        table_t table;
        table.header() << "" << "name" << "type" << "odims" << "psize" << "#kflops" << "inputs";
        table.delim();
        for (size_t i = 0; i < m_nodes.size(); ++ i)
        {
                const auto& node = m_nodes[i];
                auto& row = table.append();
                row     << (i + 1) << node.m_name << node.m_type
                        << node.m_node->odims()
                        << node.m_node->psize()
                        << node.m_probe_output.kflops()
                        << join(node_names(node.m_inodes));
        }
        table.delim();
        table.append() << "" << "" << "" << odims() << psize() << m_probe_output.kflops() << "";
        std::cout << table;
}

probes_t model_t::probes() const
{
        probes_t probes;

        const auto append = [&probes = probes] (const probe_t& probe)
        {
                if (probe)
                {
                        probes.push_back(probe);
                }
        };

        append(m_probe_output);
        append(m_probe_ginput);
        append(m_probe_gparam);

        for (const auto& node : m_nodes)
        {
                append(node.m_probe_output);
                append(node.m_probe_ginput);
                append(node.m_probe_gparam);
        }

        return probes;
}
