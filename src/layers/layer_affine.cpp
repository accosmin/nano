#include "layer_affine.h"
#include "math/numeric.h"

using namespace nano;

affine_layer_t::affine_layer_t(const string_t& params) :
        layer_t(to_params(params, "omaps", "10[1,4096]", "orows", "1[1,4096]", "ocols", "1[1,4096]"))
{
}

rlayer_t affine_layer_t::clone() const
{
        return std::make_unique<affine_layer_t>(*this);
}

void affine_layer_t::configure(const tensor3d_dims_t& idims, const string_t& name)
{
        const auto imaps = std::get<0>(idims);
        const auto irows = std::get<1>(idims);
        const auto icols = std::get<2>(idims);
        const auto omaps = nano::clamp(from_params<tensor_size_t>(config(), "omaps"), 1, 4096);
        const auto orows = nano::clamp(from_params<tensor_size_t>(config(), "orows"), 1, 4096);
        const auto ocols = nano::clamp(from_params<tensor_size_t>(config(), "ocols"), 1, 4096);

        const auto params = affine_params_t{imaps, irows, icols, omaps, orows, ocols};
        if (!params.valid())
        {
                throw std::invalid_argument("invalid configuration for the affine layer");
        }

        m_kernel = affine4d_t{params};

        m_probe_output = probe_t{name, name + "(output)", params.flops_output()};
        m_probe_ginput = probe_t{name, name + "(ginput)", params.flops_ginput()};
        m_probe_gparam = probe_t{name, name + "(gparam)", params.flops_gparam()};
}

tensor_size_t affine_layer_t::fanin() const
{
        return m_kernel.params().isize();
}

void affine_layer_t::output(const tensor4d_t& idata, const tensor1d_t& pdata, tensor4d_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_output.measure([&] () { m_kernel.output(idata, wdata(pdata), bdata(pdata), odata); }, count);
}

void affine_layer_t::ginput(tensor4d_t& idata, const tensor1d_t& pdata, const tensor4d_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_ginput.measure([&] () { m_kernel.ginput(idata, wdata(pdata), bdata(pdata), odata); }, count);
}

void affine_layer_t::gparam(const tensor4d_t& idata, tensor1d_t& pdata, const tensor4d_t& odata)
{
        const auto count = idata.size<0>();
        m_probe_gparam.measure([&] () { m_kernel.gparam(idata, wdata(pdata), bdata(pdata), odata); }, count);
}
