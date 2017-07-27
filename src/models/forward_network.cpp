#include "io/ibstream.h"
#include "io/obstream.h"
#include "math/random.h"
#include "math/numeric.h"
#include "tensor/numeric.h"
#include "text/to_string.h"
#include "text/algorithm.h"
#include "forward_network.h"
#include "logger.h"

using namespace nano;

forward_network_t::forward_network_t(const forward_network_t& other) :
        m_idims(other.m_idims),
        m_odims(other.m_odims),
        m_idata(other.m_idata),
        m_odata(other.m_odata),
        m_xdata(other.m_xdata),
        m_pdata(other.m_pdata),
        m_gdata(other.m_gdata),
        m_probe_output(other.m_probe_output),
        m_probe_ginput(other.m_probe_ginput),
        m_probe_gparam(other.m_probe_gparam)
{
        for (const auto& layer : other.m_layers)
        {
                m_layers.emplace_back(layer->clone());
        }
}

rmodel_t forward_network_t::clone() const
{
        return std::make_unique<forward_network_t>(*this);
}

bool forward_network_t::save(const string_t& path) const
{
        obstream_t ob(path);
        return  ob.write(m_idims) &&
                ob.write(m_odims) &&
                ob.write(config()) &&
                ob.write_vector(m_pdata);
}

bool forward_network_t::load(const string_t& path)
{
        string_t params;
        ibstream_t ib(path);
        return  ib.read(m_idims) &&
                ib.read(m_odims) &&
                ib.read(params) &&
                config(params) == params &&
                configure(m_idims, m_odims) &&
                ib.read_vector(m_pdata) &&
                m_pdata.size() == psize();
}

const vector_t& forward_network_t::params() const
{
        return m_pdata;
}

void forward_network_t::params(const vector_t& x)
{
        assert(x.size() == m_pdata.size());
        m_pdata = x;
}

const tensor3d_t& forward_network_t::output(const tensor3d_t& input)
{
        assert(input.dims() == idims());

        scalar_t* pxdata = m_xdata.data();
        scalar_t* ppdata = m_pdata.data();

        m_probe_output.measure([&] ()
        {
                // forward step
                map_vector(pxdata, nano::size(idims())) = input.vector();
                for (size_t l = 0; l < n_layers(); ++ l)
                {
                        auto& layer = m_layers[l];
                        layer->output(pxdata, ppdata, pxdata + layer->isize());
                        pxdata += layer->isize();
                        ppdata += layer->psize();
                }
                m_odata = map_tensor(pxdata, odims());
                pxdata += nano::size(odims());
        });

        assert(pxdata == m_xdata.data() + m_xdata.size());
        assert(ppdata == m_pdata.data() + m_pdata.size());

        return m_odata;
}

const tensor3d_t& forward_network_t::ginput(const vector_t& output)
{
        assert(output.size() == nano::size(odims()));
        assert(!m_layers.empty());

        scalar_t* pxdata = m_xdata.data() + m_xdata.size();
        scalar_t* ppdata = m_pdata.data() + m_pdata.size();

        m_probe_ginput.measure([&] ()
        {
                // backward step
                map_vector(pxdata - nano::size(odims()), nano::size(odims())) = output;
                for (size_t l = n_layers(); l > 0; -- l)
                {
                        auto& layer = m_layers[l - 1];
                        layer->ginput(pxdata - layer->xsize(), ppdata - layer->psize(), pxdata - layer->osize());
                        pxdata -= layer->osize();
                        ppdata -= layer->psize();
                }
                pxdata -= nano::size(idims());
                m_idata = map_tensor(pxdata, idims());
        });

        assert(pxdata == m_xdata.data());
        assert(ppdata == m_pdata.data());

        return m_idata;
}

const vector_t& forward_network_t::gparam(const vector_t& output)
{
        assert(output.size() == nano::size(odims()));
        assert(!m_layers.empty());

        scalar_t* pxdata = m_xdata.data() + m_xdata.size();
        scalar_t* ppdata = m_pdata.data() + m_pdata.size();
        scalar_t* pgdata = m_gdata.data() + m_gdata.size();

        m_probe_gparam.measure([&] ()
        {
                // backward step
                map_vector(pxdata - nano::size(odims()), nano::size(odims())) = output;
                for (size_t l = n_layers(); l > 0; -- l)
                {
                        auto& layer = m_layers[l - 1];
                        layer->gparam(pxdata - layer->xsize(), pgdata - layer->psize(), pxdata - layer->osize());
                        if (l > 1)
                        {
                                layer->ginput(pxdata - layer->xsize(), ppdata - layer->psize(), pxdata - layer->osize());
                        }
                        pxdata -= layer->osize();
                        ppdata -= layer->psize();
                        pgdata -= layer->psize();
                }
                pxdata -= nano::size(idims());
        });

        assert(pxdata == m_xdata.data());
        assert(ppdata == m_pdata.data());
        assert(pgdata == m_gdata.data());

        return m_gdata;
}

void forward_network_t::random()
{
        scalar_t* ppdata = m_pdata.data();

        for (const auto& layer : m_layers)
        {
                const auto div = static_cast<scalar_t>(layer->fanin());
                const auto min = -std::sqrt(6 / (1 + div));
                const auto max = +std::sqrt(6 / (1 + div));

                nano::set_random(random_t<scalar_t>(min, max), map_vector(ppdata, layer->psize()));
                ppdata += layer->psize();
        }

        assert(ppdata == m_pdata.data() + m_pdata.size());
}

bool forward_network_t::configure(const tensor3d_dims_t& idims, const tensor3d_dims_t& odims)
{
        m_idims = idims;
        m_odims = odims;

        auto xdims = idims;
        auto xsize = nano::size(idims);
        auto psize = tensor_size_t(0);

        m_layers.clear();

        int64_t flops_output = 0;
        int64_t flops_ginput = 0;
        int64_t flops_gparam = 0;

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
                        throw std::invalid_argument("invalid layer description");
                }

                const auto layer_id = layer_tokens[0];
                const auto layer_params = layer_tokens.size() == 2 ? layer_tokens[1] : string_t();
                const auto layer_name = "[" + align(to_string(l + 1), 2, alignment::right, '0') + ":" +
                        align(layer_id, 10, alignment::left, '.') + "]";

                auto layer = nano::get_layers().get(layer_id, layer_params);
                layer->configure(xdims, layer_name);

                xsize += nano::size(layer->odims());
                psize += layer->psize();
                xdims = layer->odims();

                flops_output = layer->probe_output().flops();
                flops_ginput = layer->probe_ginput().flops();
                flops_gparam = layer->probe_gparam().flops();

                m_layers.emplace_back(std::move(layer));
        }

        // check output size to match the target
        if (xdims != odims)
        {
                log_error() << "forward network: miss-matching output size " << xdims << ", expecting " << odims << "!";
                throw std::invalid_argument("invalid output layer description");
        }

        // allocate buffers
        m_idata.resize(idims);
        m_odata.resize(odims);
        m_xdata.resize(xsize);
        m_pdata.resize(psize);
        m_gdata.resize(psize);

        m_pdata.setZero();

        // setup probes
        m_probe_output = probe_t{"network", "network(output)", flops_output};
        m_probe_ginput = probe_t{"network", "network(ginput)", flops_ginput};
        m_probe_gparam = probe_t{"network", "network(gparam)", flops_gparam};

        return true;
}

tensor3d_dims_t forward_network_t::idims() const
{
        return m_idims;
}

tensor3d_dims_t forward_network_t::odims() const
{
        return m_odims;
}

void forward_network_t::describe() const
{
        const scalar_t* ppdata = m_pdata.data();

        log_info() << "forward network [" << config() << "]";
        for (size_t l = 0; l < n_layers(); ++ l)
        {
                const auto& layer = m_layers[l];

                // collect & print statistics for its parameters / layer
                const auto param = map_vector(ppdata, layer->psize());
                ppdata += layer->psize();

                const auto flops_output = idiv(layer->probe_output().flops(), 1024);
                const auto flops_ginput = idiv(layer->probe_ginput().flops(), 1024);
                const auto flops_gparam = idiv(layer->probe_gparam().flops(), 1024);

                if (layer->psize())
                {
                        const auto min = param.minCoeff();
                        const auto max = param.maxCoeff();

                        log_info()
                                << "forward network " << layer->probe_output().basename()
                                << ": in(" << layer->idims() << ") -> " << "out(" << layer->odims() << ")"
                                << ", kFLOPs = {" << flops_output << "," << flops_ginput << "," << flops_gparam << "}"
                                << ", params = " << layer->psize()
                                << ", range = [" << min << ", " << max << "].";
                }
                else
                {
                        log_info()
                                << "forward network " << layer->probe_output().basename()
                                << ": in(" << layer->idims() << ") -> " << "out(" << layer->odims() << ")"
                                << ", kFLOPs = {" << flops_output << "," << flops_ginput << "," << flops_gparam << "}.";
                }
        }
        log_info() << "forward network: parameters = " << psize() << ".";
}

tensor_size_t forward_network_t::psize() const
{
        return m_gdata.size();
}

probes_t forward_network_t::probes() const
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

        for (const auto& layer : m_layers)
        {
                append(layer->probe_output());
                append(layer->probe_ginput());
                append(layer->probe_gparam());
        }

        return probes;
}
