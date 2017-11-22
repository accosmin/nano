#include "layer_conv3d.h"

using namespace nano;

json_reader_t& conv3d_layer_t::config(json_reader_t& reader)
{
        return reader.object("omaps", m_params.m_omaps, "krows", m_params.m_krows, "kcols", m_params.m_kcols,
                "kconn", m_params.m_kconn, "kdrow", m_params.m_kdrow, "kdcol", m_params.m_kdcol);
}

json_writer_t& conv3d_layer_t::config(json_writer_t& writer) const
{
        return writer.object("omaps", m_params.m_omaps, "krows", m_params.m_krows, "kcols", m_params.m_kcols,
                "kconn", m_params.m_kconn, "kdrow", m_params.m_kdrow, "kdcol", m_params.m_kdcol);
}

rlayer_t conv3d_layer_t::clone() const
{
        return std::make_unique<conv3d_layer_t>(*this);
}

bool conv3d_layer_t::resize(const tensor3d_dims_t& idims, const string_t& name)
{
        m_params.m_imaps = std::get<0>(idims);
        m_params.m_irows = std::get<1>(idims);
        m_params.m_icols = std::get<2>(idims);

        if (!m_params.valid())
        {
                return false;
        }

        m_kernel = conv4d_t{m_params};

        m_probe_output = probe_t{name, name + "(output)", m_params.flops_output()};
        m_probe_ginput = probe_t{name, name + "(ginput)", m_params.flops_ginput()};
        m_probe_gparam = probe_t{name, name + "(gparam)", m_params.flops_gparam()};
        return true;
}

tensor_size_t conv3d_layer_t::fanin() const
{
        return m_params.krows() * m_params.kcols() * m_params.imaps() / m_params.kconn();
}

void conv3d_layer_t::output(const tensor4d_cmap_t& idata, const vector_cmap_t& pdata, tensor4d_map_t&& odata)
{
        const auto count = idata.size<0>();
        m_probe_output.measure([&] () { m_kernel.output(idata, kdata(pdata), bdata(pdata), odata); }, count);
}

void conv3d_layer_t::ginput(tensor4d_map_t&& idata, const vector_cmap_t& pdata, const tensor4d_cmap_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_ginput.measure([&] () { m_kernel.ginput(idata, kdata(pdata), bdata(pdata), odata); }, count);
}

void conv3d_layer_t::gparam(const tensor4d_cmap_t& idata, vector_map_t&& pdata, const tensor4d_cmap_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_gparam.measure([&] () { m_kernel.gparam(idata, kdata(pdata), bdata(pdata), odata); }, count);
}
