#include "model.h"
#include "logger.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include "math/random.h"
#include "math/numeric.h"
#include "tensor/numeric.h"
#include "text/algorithm.h"

using namespace nano;

model_t::model_t(const model_t& other) :
        m_names(other.m_names),
        m_graph(other.m_graph),
        m_gdata(other.m_gdata),
        m_probe_output(other.m_probe_output),
        m_probe_ginput(other.m_probe_ginput),
        m_probe_gparam(other.m_probe_gparam)
{
        for (const auto& layer : other.m_nodes)
        {
                m_nodes.emplace_back(layer->clone());
        }
}

rmodel_t model_t::clone() const
{
        return std::make_unique<model_t>(*this);
}

void model_t::clear()
{
        m_names.clear();
        m_nodes.clear();
        m_graph.clear();
}

bool model_t::add(const string_t& name, const string_t& type, json_reader_t& reader)
{
        log_info() << "model: adding node [" << name << "] of type [" << type << "]...";

        if (std::find(m_names.begin(), m_names.end(), name) != m_names.end())
        {
                log_error() << "model: duplicated name!";
                return false;
        }

        auto node = get_layers().get(type);
        if (!node)
        {
                log_error() << "model: invalid node type!";
                return false;
        }

        node->config(reader);
        m_names.push_back(name);
        m_types.push_back(type);
        m_nodes.emplace_back(std::move(node));
        return true;
}

bool model_t::add(const string_t& name, const string_t& type, const string_t& json)
{
        json_reader_t reader(json);
        return add(name, type, reader);
}

bool model_t::connect(const string_t& name1, const string_t& name2)
{
        log_info() << "model: connecting nodes " << name1 << " -> " << name2 << "...";

        const auto it1 = std::find(m_names.begin(), m_names.end(), name1);
        const auto it2 = std::find(m_names.begin(), m_names.end(), name2);

        if (it1 == m_names.end() || it2 == m_names.end())
        {
                log_error() << "model: unknown node name(s)!";
                return false;
        }

        m_graph.edge(
                static_cast<dindex_t>(std::distance(m_names.begin(), it1)),
                static_cast<dindex_t>(std::distance(m_names.begin(), it2)));

        if (    m_graph.vertices() >= std::numeric_limits<dindex_t>::max() ||
                m_graph.vertices() > static_cast<dindex_t>(m_nodes.size()))
        {
                log_error() << "model: overflow detected in the computation graph!";
                return false;
        }

        return true;
}

namespace
{
        enum json_mode
        {
                none,
                nodes,
                model,
                node_name,
                node_type,
                node_config
        };

        static const std::vector<std::pair<string_t, json_mode>> tokens2mode =
        {
                { "nodes", json_mode::nodes },
                { "model", json_mode::model },
                { "name", json_mode::node_name },
                { "type", json_mode::node_type },
                { "config", json_mode::node_config }
        };

        void handle_error(const json_reader_t& reader, const char* error_message)
        {
                log_error() << error_message << " @json..." << reader.str().substr(reader.pos(), 16) << "..." << to_string(reader.tag()) << "!";
        }

        auto handle_name(const char* token_name, const size_t token_size)
        {
                const auto it = std::find_if(tokens2mode.begin(), tokens2mode.end(), [&] (const auto& tm)
                {
                        return  tm.first.size() == token_size &&
                                tm.first.compare(0, token_size, token_name, token_size) == 0;
                });
                return it == tokens2mode.end() ? json_mode::none : it->second;
        }
}

bool model_t::config(json_reader_t& reader)
{
        log_info() << "model: configuring...";

        clear();
        for (auto itend = reader.end(); reader != itend; )
        {
                const auto token = *reader;
                const auto token_name = std::get<0>(token);
                const auto token_size = std::get<1>(token);

                switch (std::get<2>(token))
                {
                case json_tag::new_object:
                        // continue reading
                        ++ reader;
                        break;

                case json_tag::end_object:
                        // done as all the other end_object tags should have been read by ::config_nodes
                        return done();

                case json_tag::name:
                        // part detected
                        switch (handle_name(token_name, token_size))
                        {
                        case json_mode::nodes:
                                if (!config_nodes(++ reader))
                                {
                                        return false;
                                }
                                break;

                        case json_mode::model:
                                if (!config_model(++ reader))
                                {
                                        return false;
                                }
                                break;

                        default:
                                handle_error(reader, "model: unexpected name");
                                return false;
                        }
                        break;

                default:
                        handle_error(reader, "model: unexpected token");
                        return false;
                }
        }

        handle_error(reader, "model: unexpected ending");
        return false;
}

bool model_t::config_nodes(json_reader_t& reader)
{
        log_info() << "model: configuring the computation nodes...";

        string_t last_name, last_type;
        json_mode last_mode = json_mode::none;

        for (auto itend = reader.end(); reader != itend; )
        {
                const auto token = *reader;
                const auto token_name = std::get<0>(token);
                const auto token_size = std::get<1>(token);

                switch (std::get<2>(token))
                {
                case json_tag::new_object:
                        if (last_mode != json_mode::node_config)
                        {
                                ++ reader;
                        }
                        else if (!add(last_name, last_type, reader))
                        {
                                return false;
                        }
                        else
                        {
                                last_mode = json_mode::none;
                        }
                        break;

                case json_tag::name:
                        switch (last_mode = handle_name(token_name, token_size))
                        {
                        case json_mode::node_name:   ++ reader; break;
                        case json_mode::node_type:   ++ reader; break;
                        case json_mode::node_config: ++ reader; break;
                        default:                     handle_error(reader, "model: unexpected nodes name"); return false;
                        }
                        break;

                case json_tag::value:
                        switch (last_mode)
                        {
                        case json_mode::node_name:   last_name.assign(token_name, token_size); ++ reader; break;
                        case json_mode::node_type:   last_type.assign(token_name, token_size); ++ reader; break;
                        default:                     handle_error(reader, "model: unexpected nodes value"); return false;
                        }
                        break;

                case json_tag::new_array:
                        // go on, nodes starting
                case json_tag::end_object:
                        // go on, next node
                        ++ reader;
                        break;

                case json_tag::end_array:
                        // done with all nodes
                        ++ reader;
                        return true;

                default:
                        handle_error(reader, "model: unexpected nodes token");
                        return false;
                }
        }

        return false;
}

bool model_t::config_model(json_reader_t& reader)
{
        log_info() << "model: configuring the computation graph...";

        string_t last_name;
        for (auto itend = reader.end(); reader != itend; )
        {
                const auto token = *reader;
                const auto token_name = std::get<0>(token);
                const auto token_size = std::get<1>(token);

                switch (std::get<2>(token))
                {
                case json_tag::new_array:
                case json_tag::end_array:
                        last_name.clear();
                        ++ reader;
                        break;

                case json_tag::value:
                        if (!last_name.empty() && !connect(last_name, string_t(token_name, token_size)))
                        {
                                return false;
                        }
                        last_name.assign(token_name, token_size);
                        ++ reader;
                        break;

                case json_tag::end_object:
                        // done
                        return true;

                default:
                        handle_error(reader, "model: unexpected model token");
                        return false;
                }
        }

        return false;
}

bool model_t::done()
{
        log_info() << "model: checking the computation graph...";

        m_graph.done();

        if (m_graph.vertices() < 1)
        {
                log_error() << "model: expecting at least a node!";
                return false;
        }

        const auto sources = m_graph.sources();
        if (sources.size() != 1)
        {
                log_error() << "model: expecting exactly one input node!";
                return false;
        }

        const auto sinks = m_graph.sinks();
        if (sinks.size() != 1)
        {
                // todo: may relax this condition (many outputs may be needed to solve reinforcement learning problems)
                log_error() << "model: expecting exactly one output node!";
                return false;
        }

        if (!m_graph.dag())
        {
                // todo: may relax this condition
                log_error() << "model: cyclic computation graphs are not supported!";
                return false;
        }

        for (const auto sink : sinks)
        {
                const auto it = std::find_if(sources.begin(), sources.end(), [&] (const auto source)
                {
                        return m_graph.connected(source, sink);
                });

                if (it == sources.end())
                {
                        log_error() << "model: detected unreachable output node [" << m_names[sink] << "]!";
                        return false;
                }
        }

        for (const auto source : sources)
        {
                const auto it = std::find_if(sinks.begin(), sinks.end(), [&] (const auto sink)
                {
                        return m_graph.connected(source, sink);
                });

                if (it == sinks.end())
                {
                        log_error() << "model: detected unused input node [" << m_names[source] << "]!";
                        return false;
                }
        }

        return true;
}

bool model_t::config(const string_t& json)
{
        json_reader_t reader(json);
        return config(reader);
}

void model_t::config(json_writer_t& writer) const
{
        writer.new_object().name("nodes").new_array();
        for (size_t i = 0; i < m_names.size(); ++ i)
        {
                const auto& name = m_names[i];
                const auto& type = m_types[i];
                const auto& node = m_nodes[i];

                writer.new_object();
                        writer.pairs("name", name, "type", type).next();
                        writer.name("config"); node->config(writer);
                writer.end_object();
                if (i + 1 < m_names.size())
                {
                        writer.next();
                }
        }
        writer.end_array().next();
        writer.name("model").new_array();
        for (size_t i = 0; i < m_graph.edges().size(); ++ i)
        {
                const auto& edge = m_graph.edges()[i];

                assert(edge.first < m_names.size());
                assert(edge.second < m_names.size());
                const auto& name1 = m_names[edge.first];
                const auto& name2 = m_names[edge.second];

                writer.array(name1, name2);
                if (i + 1 < m_graph.edges().size())
                {
                        writer.next();
                }
        }
        writer.end_array().end_object();
}

bool model_t::save(const string_t& path) const
{
        NANO_UNUSED1(path);
        return false;
        /*
        obstream_t ob(path);
        return  ob.write(idims()) &&
                ob.write(odims()) &&
                ob.write(config()) &&
                ob.write_vector(params());*/
}

bool model_t::load(const string_t& path)
{
        NANO_UNUSED1(path);
        return false;
        /*
        tensor3d_dims_t idims, odims;
        vector_t pdata;
        string_t param;

        ibstream_t ib(path);
        return  ib.read(idims) &&
                ib.read(odims) &&
                ib.read(param) &&
                ib.read_vector(pdata) &&
                config(param) == param &&
                config(idims, odims) &&
                pdata.size() == psize() &&
                [&] () { params(pdata); return true; }();
        */
}

vector_t model_t::params() const
{
        vector_t pdata(psize());

        tensor_size_t pindex = 0;
        for (const auto& layer : m_nodes)
        {
                if (layer->psize())
                {
                        assert(pindex + layer->psize() <= pdata.size());
                        map_vector(pdata.data() + pindex, layer->psize()) = layer->param().vector();
                        pindex += layer->psize();
                }
        }
        assert(pindex == psize());

        return pdata;
}

void model_t::params(const vector_t& pdata)
{
        assert(pdata.size() == psize());

        tensor_size_t pindex = 0;
        for (const auto& layer : m_nodes)
        {
                if (layer->psize())
                {
                        assert(pindex + layer->psize() <= pdata.size());
                        layer->param(map_tensor(pdata.data() + pindex, layer->pdims()));
                        pindex += layer->psize();
                }
        }
        assert(pindex == psize());
}

const tensor4d_t& model_t::output(const tensor4d_t& idata)
{
        assert(idata.tensor(0).dims() == idims());

        auto pidata = &idata;

        const auto count = idata.size<0>();
        m_probe_output.measure([&] ()
        {
                // forward step
                for (const auto& layer : m_nodes)
                {
                        pidata = &layer->output(*pidata);
                }
        }, count);

        return *pidata;
}

const tensor4d_t& model_t::ginput(const tensor4d_t& odata)
{
        assert(odata.tensor(0).dims() == odims());

        auto podata = &odata;

        const auto count = odata.size<0>();
        m_probe_ginput.measure([&] ()
        {
                // backward step
                for (auto it = m_nodes.rbegin(); it != m_nodes.rend(); ++ it)
                {
                        podata = &(*it)->ginput(*podata);
                }
        }, count);

        return *podata;
}

const tensor1d_t& model_t::gparam(const tensor4d_t& odata)
{
        assert(odata.tensor(0).dims() == odims());

        auto podata = &odata;
        tensor_size_t pindex = m_gdata.size();

        const auto count = odata.size<0>();
        m_probe_gparam.measure([&] ()
        {
                // backward step
                for (size_t l = m_nodes.size(); l > 0; -- l)
                {
                        auto& layer = m_nodes[l - 1];
                        const auto& gdata = layer->gparam(*podata);
                        map_vector(m_gdata.data() + pindex - layer->psize(), layer->psize()) = gdata.vector();
                        if (l > 1)
                        {
                                podata = &layer->ginput(*podata);
                        }
                        pindex -= layer->psize();
                }
        }, count);
        assert(pindex == 0);

        return m_gdata;
}

void model_t::random()
{
        tensor_size_t pindex = 0;
        for (const auto& layer : m_nodes)
        {
                if (layer->psize())
                {
                        const auto div = static_cast<scalar_t>(layer->fanin());
                        const auto min = -std::sqrt(6 / (1 + div));
                        const auto max = +std::sqrt(6 / (1 + div));

                        auto gdata = map_tensor(m_gdata.data() + pindex, layer->pdims());
                        auto pdata = map_tensor(static_cast<const scalar_t*>(m_gdata.data()) + pindex, layer->pdims());

                        nano::set_random(random_t<scalar_t>(min, max), gdata);
                        layer->param(pdata);

                        pindex += layer->psize();
                }
        }
        assert(pindex == psize());
}

bool model_t::resize(const tensor3d_dims_t& idims, const tensor3d_dims_t& odims)
{
        return false;

        auto xdims = idims;

        tensor_size_t psize = 0;
        int64_t flops_output = 0;
        int64_t flops_ginput = 0;
        int64_t flops_gparam = 0;

        /*
        // create layers
        const auto net_params = nano::split(config(), ";");
        for (size_t l = 0; l < net_params.size(); ++ l)
        {
                if (net_params[l].size() <= 1)
                {
                        continue;
                }

                const auto layer_tokens = nano::split(net_params[l], ":");
                if (layer_tokens.size() != 2 && layer_tokens.size() != 1)
                {
                        log_error() << "forward network: invalid layer description <"
                                    << net_params[l] << ">! expecting <layer_id[:layer_parameters]>!";
                        return false;
                }

                const auto layer_id = layer_tokens[0];
                const auto layer_params = layer_tokens.size() == 2 ? layer_tokens[1] : string_t();
                const auto layer_name = "[" + align(to_string(l + 1), 2, alignment::right, '0') + ":" +
                        align(layer_id, 10, alignment::left, '.') + "]";

                auto layer = nano::get_layers().get(layer_id, layer_params);
                if (!layer->config(xdims, layer_name))
                {
                        log_error() << "forward network: invalid layer configuration <"
                                    << net_params[l] << ">!";
                        return false;
                }

                xdims = layer->odims();
                psize += layer->psize();

                flops_output += layer->probe_output().flops();
                flops_ginput += layer->probe_ginput().flops();
                flops_gparam += layer->probe_gparam().flops();

                m_nodes.emplace_back(std::move(layer));
        }*/

        // check output size to match the target
        if (xdims != odims)
        {
                log_error() << "model: miss-matching output size " << xdims << ", expecting " << odims << "!";
                return false;
        }

        // allocate buffers
        m_gdata.resize(psize);

        // setup probes
        m_probe_output = probe_t{"model", "model(output)", flops_output};
        m_probe_ginput = probe_t{"model", "model(ginput)", flops_ginput};
        m_probe_gparam = probe_t{"model", "model(gparam)", flops_gparam};

        return true;
}

tensor_size_t model_t::psize() const
{
        return m_gdata.size();
}

tensor3d_dims_t model_t::idims() const
{
        return m_nodes.at(0)->idims();
}

tensor3d_dims_t model_t::odims() const
{
        return m_nodes.at(m_nodes.size() - 1)->odims();
}

void model_t::describe() const
{
        /*log_info() << "forward network [" << config() << "]";
        for (const auto& layer : m_nodes)
        {
                log_info()
                        << "forward network " << layer->probe_output().basename()
                        << ": idims = " << layer->idims()
                        << ", odims = " << layer->odims()
                        << ", pdims = " << layer->pdims() << ".";
        }
        log_info() << "forward network: parameters = " << psize() << ".";*/
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
                append(node->probe_output());
                append(node->probe_ginput());
                append(node->probe_gparam());
        }

        return probes;
}
