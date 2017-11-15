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

bool model_t::add(const string_t& type, const string_t& name, json_reader_t& reader)
{
        log_info() << "model: adding node " << name << " of type " << type << "...";

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
        m_nodes.emplace_back(std::move(node));
        return true;
}

bool model_t::add(const string_t& type, const string_t& name, const string_t& json)
{
        json_reader_t reader(json);
        return add(type, name, reader);
}

bool model_t::connect(const string_t& name1, const string_t& name2)
{
        log_info() << "model: connecting nodes " << name1 << " -> " << name2 << "...";

        const auto it1 = std::find(m_names.begin(), m_names.end(), name1);
        const auto it2 = std::find(m_names.begin(), m_names.end(), name2);

        if (it1 == m_names.end() || it2 == m_names.end())
        {
                log_error() << "model: unknown name(s)!";
                return false;
        }

        if (!m_graph.edge(
                static_cast<dindex_t>(std::distance(m_names.begin(), it1)),
                static_cast<dindex_t>(std::distance(m_names.begin(), it2))))
        {
                log_error() << "model: duplicated edge!";
                return false;
        }

        if (    m_graph.vertices() >= std::numeric_limits<dindex_t>::max() ||
                m_graph.vertices() > static_cast<dindex_t>(m_nodes.size()))
        {
                log_error() << "model: overflow detected in the computation graph!";
                return false;
        }

        return true;
}

bool model_t::config(json_reader_t& reader)
{
        clear();

//        const char* const token_nodes = "nodes";
  //      const char* const token_model = "model";
   //     const char* const token_node_name = "name";
    //    const char* const token_node_type = "type";
     //   const char* const token_node_config = "config";

        NANO_UNUSED1(reader);

        // todo

        return false;
}

bool model_t::config(const string_t& json)
{
        json_reader_t reader(json);
        return config(reader);
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
