#include "layer_affine.h"

using namespace nano;

json_reader_t& affine_layer_t::config(json_reader_t& reader)
{
        return reader.object("omaps", m_params.m_omaps, "orows", m_params.m_orows, "ocols", m_params.m_ocols);
}

json_writer_t& affine_layer_t::config(json_writer_t& writer) const
{
        return writer.object("omaps", m_params.m_omaps, "orows", m_params.m_orows, "ocols", m_params.m_ocols);
}

rlayer_t affine_layer_t::clone() const
{
        return std::make_unique<affine_layer_t>(*this);
}

bool affine_layer_t::resize(const tensor3d_dims_t& idims, const string_t& name)
{
        m_params.m_imaps = std::get<0>(idims);
        m_params.m_irows = std::get<1>(idims);
        m_params.m_icols = std::get<2>(idims);

        if (!m_params.valid())
        {
                return false;
        }

        m_kernel = affine4d_t{m_params};

        m_probe_output = probe_t{name, name + "(output)", m_params.flops_output()};
        m_probe_ginput = probe_t{name, name + "(ginput)", m_params.flops_ginput()};
        m_probe_gparam = probe_t{name, name + "(gparam)", m_params.flops_gparam()};
        return true;
}

tensor_size_t affine_layer_t::fanin() const
{
        return m_params.isize();
}

void affine_layer_t::output(const tensor4d_cmap_t& idata, const vector_cmap_t& pdata, tensor4d_map_t&& odata)
{
        const auto count = idata.size<0>();
        m_probe_output.measure([&] () { m_kernel.output(idata, wdata(pdata), bdata(pdata), odata); }, count);
}

void affine_layer_t::ginput(tensor4d_map_t&& idata, const vector_cmap_t& pdata, const tensor4d_cmap_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_ginput.measure([&] () { m_kernel.ginput(idata, wdata(pdata), bdata(pdata), odata); }, count);
}

void affine_layer_t::gparam(const tensor4d_cmap_t& idata, vector_map_t&& pdata, const tensor4d_cmap_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_gparam.measure([&] () { m_kernel.gparam(idata, wdata(pdata), bdata(pdata), odata); }, count);
}
