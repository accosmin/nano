#include "math/numeric.h"
#include "layer_convolution.h"

using namespace nano;

json_reader_t& convolution_layer_t::config(json_reader_t& reader)
{
        return reader.object("omaps", m_params.m_omaps, "krows", m_params.m_krows, "kcols", m_params.m_kcols,
                "kconn", m_params.m_kconn, "kdrow", m_params.m_kdrow, "kdcol", m_params.m_kdcol);
}

json_writer_t& convolution_layer_t::config(json_writer_t& writer) const
{
        return writer.object("omaps", m_params.m_omaps, "krows", m_params.m_krows, "kcols", m_params.m_kcols,
                "kconn", m_params.m_kconn, "kdrow", m_params.m_kdrow, "kdcol", m_params.m_kdcol);
}

rlayer_t convolution_layer_t::clone() const
{
        return std::make_unique<convolution_layer_t>(*this);
}

bool convolution_layer_t::config(const tensor3d_dims_t& idims, const string_t& name)
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

tensor_size_t convolution_layer_t::fanin() const
{
        return m_params.krows() * m_params.kcols() * m_params.imaps() / m_params.kconn();
}

void convolution_layer_t::output(const tensor4d_t& idata, const tensor1d_t& pdata, tensor4d_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_output.measure([&] () { m_kernel.output(idata, kdata(pdata), bdata(pdata), odata); }, count);
}

void convolution_layer_t::ginput(tensor4d_t& idata, const tensor1d_t& pdata, const tensor4d_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_ginput.measure([&] () { m_kernel.ginput(idata, kdata(pdata), bdata(pdata), odata); }, count);
}

void convolution_layer_t::gparam(const tensor4d_t& idata, tensor1d_t& pdata, const tensor4d_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_gparam.measure([&] () { m_kernel.gparam(idata, kdata(pdata), bdata(pdata), odata); }, count);
}
