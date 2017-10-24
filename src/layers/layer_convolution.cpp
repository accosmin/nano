#include "math/numeric.h"
#include "layer_convolution.h"

using namespace nano;

convolution_layer_t::convolution_layer_t(const string_t& params) :
        layer_t(to_params(params, "omaps", "16[1,4096]", "krows", "8[1,32]", "kcols", "8[1,32]",
        "kconn", "1[1,16]", "kdrow", "1[1,8]", "kdcol", "1[1,8]"))
{
}

rlayer_t convolution_layer_t::clone() const
{
        return std::make_unique<convolution_layer_t>(*this);
}

bool convolution_layer_t::configure(const tensor3d_dims_t& idims, const string_t& name)
{
        const auto imaps = std::get<0>(idims);
        const auto irows = std::get<1>(idims);
        const auto icols = std::get<2>(idims);

        const auto omaps = clamp(from_params<tensor_size_t>(config(), "omaps"), 1, 4096);
        const auto krows = clamp(from_params<tensor_size_t>(config(), "krows"), 1, 32);
        const auto kcols = clamp(from_params<tensor_size_t>(config(), "kcols"), 1, 32);
        const auto kconn = clamp(from_params<tensor_size_t>(config(), "kconn"), 1, 16);
        const auto kdrow = clamp(from_params<tensor_size_t>(config(), "kdrow"), 1, 8);
        const auto kdcol = clamp(from_params<tensor_size_t>(config(), "kdcol"), 1, 8);

        const auto params = conv_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};
        if (!params.valid())
        {
                return false;
        }

        m_kernel = conv4d_t{params};

        m_probe_output = probe_t{name, name + "(output)", params.flops_output()};
        m_probe_ginput = probe_t{name, name + "(ginput)", params.flops_ginput()};
        m_probe_gparam = probe_t{name, name + "(gparam)", params.flops_gparam()};
        return true;
}

tensor_size_t convolution_layer_t::fanin() const
{
        const auto& params = m_kernel.params();
        return params.krows() * params.kcols() * params.imaps() / params.kconn();
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
