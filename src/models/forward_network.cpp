#include "timer.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include "math/random.h"
#include "math/numeric.h"
#include "tensor/numeric.h"
#include "text/to_string.h"
#include "text/algorithm.h"
#include "forward_network.h"
#include "logger.h"

namespace nano
{
        forward_network_t::layer_info_t::layer_info_t(const string_t& name, rlayer_t layer) :
                m_name(name), m_layer(std::move(layer))
        {
        }

        forward_network_t::layer_info_t::layer_info_t(const layer_info_t& other) :
                m_name(other.m_name),
                m_layer(other.m_layer->clone()),
                m_output_timings(other.m_output_timings),
                m_ginput_timings(other.m_ginput_timings),
                m_gparam_timings(other.m_gparam_timings)
        {
        }

        void forward_network_t::layer_info_t::output(scalar_t* idata, scalar_t* param, scalar_t* odata)
        {
                const timer_t timer;
                m_layer->output(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                m_output_timings(static_cast<size_t>(timer.microseconds().count()));
        }

        void forward_network_t::layer_info_t::ginput(scalar_t* idata, scalar_t* param, scalar_t* odata)
        {
                const timer_t timer;
                m_layer->ginput(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                m_ginput_timings(static_cast<size_t>(timer.microseconds().count()));
        }

        void forward_network_t::layer_info_t::gparam(scalar_t* idata, scalar_t* param, scalar_t* odata)
        {
                const timer_t timer;
                m_layer->gparam(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                m_gparam_timings(static_cast<size_t>(timer.microseconds().count()));
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
                        ob.write(m_configuration) &&
                        ob.write_vector(m_pdata);
        }

        bool forward_network_t::load(const string_t& path)
        {
                ibstream_t ib(path);
                return  ib.read(m_idims) &&
                        ib.read(m_odims) &&
                        ib.read(m_configuration) &&
                        configure(m_idims, m_odims) &&
                        ib.read_vector(m_pdata) &&
                        m_pdata.size() == psize();
        }

        bool forward_network_t::save(vector_t& x) const
        {
                return (x = m_pdata).size() == psize();
        }

        bool forward_network_t::load(const vector_t& x)
        {
                return (x.size() == psize()) && (m_pdata = x).size() == psize();
        }

        const tensor3d_t& forward_network_t::output(const tensor3d_t& input)
        {
                assert(input.dims() == idims());

                scalar_t* pxdata = m_xdata.data();
                scalar_t* ppdata = m_pdata.data();

                // forward step
                map_vector(pxdata, nano::size(idims())) = input.vector();
                for (size_t l = 0; l < n_layers(); ++ l)
                {
                        auto& layer = m_layers[l];
                        layer.output(pxdata, ppdata, pxdata + layer.isize());
                        pxdata += layer.isize();
                        ppdata += layer.psize();
                }
                m_odata = map_tensor(pxdata, odims());
                pxdata += nano::size(odims());

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

                // backward step
                map_vector(pxdata - nano::size(odims()), nano::size(odims())) = output;
                for (size_t l = n_layers(); l > 0; -- l)
                {
                        auto& layer = m_layers[l - 1];
                        layer.ginput(pxdata - layer.xsize(), ppdata - layer.psize(), pxdata - layer.osize());
                        pxdata -= layer.osize();
                        ppdata -= layer.psize();
                }
                pxdata -= nano::size(idims());
                m_idata = map_tensor(pxdata, idims());

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

                // backward step
                map_vector(pxdata - nano::size(odims()), nano::size(odims())) = output;
                for (size_t l = n_layers(); l > 0; -- l)
                {
                        auto& layer = m_layers[l - 1];
                        layer.gparam(pxdata - layer.xsize(), pgdata - layer.psize(), pxdata - layer.osize());
                        if (l > 1)
                        {
                                layer.ginput(pxdata - layer.xsize(), ppdata - layer.psize(), pxdata - layer.osize());
                        }
                        pxdata -= layer.osize();
                        ppdata -= layer.psize();
                        pgdata -= layer.psize();
                }
                pxdata -= nano::size(idims());

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
                        const auto div = static_cast<scalar_t>(layer.isize() + layer.osize());
                        const auto min = -std::sqrt(6 / (1 + div));
                        const auto max = +std::sqrt(6 / (1 + div));

                        nano::set_random(random_t<scalar_t>(min, max), map_vector(ppdata, layer.psize()));
                        ppdata += layer.psize();
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
                        layer->configure(xdims);

                        xsize += nano::size(layer->odims());
                        psize += layer->psize();
                        xdims = layer->odims();

                        m_layers.emplace_back(layer_name, std::move(layer));
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
                        const auto param = map_vector(ppdata, layer.psize());
                        ppdata += layer.psize();

                        if (layer.psize())
                        {
                                const auto l2 = param.lpNorm<2>();
                                const auto li = param.lpNorm<Eigen::Infinity>();

                                log_info()
                                        << "forward network " << layer.m_name
                                        << ": in(" << layer.idims() << ") -> " << "out(" << layer.odims() << ")"
                                        << ", kFLOPs = " << nano::idiv(layer.flops(), 1024)
                                        << ", params = " << layer.psize() << " [L2=" << l2 << ", Li=" << li << "].";
                        }
                        else
                        {
                                log_info()
                                        << "forward network " << layer.m_name
                                        << ": in(" << layer.idims() << ") -> " << "out(" << layer.odims() << ")"
                                        << ", kFLOPs = " << nano::idiv(layer.flops(), 1024) << ".";
                        }
                }
                log_info() << "forward network: parameters = " << psize() << ".";
        }

        tensor_size_t forward_network_t::psize() const
        {
                return m_gdata.size();
        }

        model_t::timings_t forward_network_t::timings() const
        {
                model_t::timings_t ret;
                for (const auto& layer : m_layers)
                {
                        if (layer.m_output_timings.count() > 1) ret[layer.m_name + " (output)"] = layer.m_output_timings;
                        if (layer.m_ginput_timings.count() > 1) ret[layer.m_name + " (ginput)"] = layer.m_ginput_timings;
                        if (layer.m_gparam_timings.count() > 1) ret[layer.m_name + " (gparam)"] = layer.m_gparam_timings;
                }

                return ret;
        }
}
