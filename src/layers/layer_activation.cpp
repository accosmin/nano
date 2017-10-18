#include "math/numeric.h"
#include "layer_activation.h"

using namespace nano;

activation_layer_t::activation_layer_t(const string_t& params) :
        layer_t(params),
        m_idims({0, 0, 0}),
        m_odims({0, 0, 0})
{
}

void activation_layer_t::configure(const tensor3d_dims_t& idims, const string_t& name)
{
        m_idims = idims;
        m_odims = idims;

        m_probe_output = probe_t{name, name + "(output)", 10 * isize()};
        m_probe_ginput = probe_t{name, name + "(ginput)", 10 * isize()};
        m_probe_gparam = probe_t{name, name + "(gparam)", 0};
}

void activation_layer_t::output(tensor3d_cmap_t idata, tensor1d_cmap_t param, tensor3d_map_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());
        NANO_UNUSED1_RELEASE(param);

        m_probe_output.measure([&] ()
        {
                aoutput(idata.array(), odata.array());
        });
}

void activation_layer_t::ginput(tensor3d_map_t idata, tensor1d_cmap_t param, tensor3d_cmap_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());
        NANO_UNUSED1_RELEASE(param);

        m_probe_ginput.measure([&] ()
        {
                aginput(idata.array(), odata.array());
        });
}

void activation_layer_t::gparam(tensor3d_cmap_t idata, tensor1d_map_t param, tensor3d_cmap_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());
        NANO_UNUSED3_RELEASE(idata, param, odata);
}

rlayer_t activation_layer_sine_t::clone() const
{
        return std::make_unique<activation_layer_sine_t>(*this);
}

void activation_layer_sine_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
{
        odata = idata.sin();
}

void activation_layer_sine_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
{
        idata = odata * idata.cos();
}

rlayer_t activation_layer_snorm_t::clone() const
{
        return std::make_unique<activation_layer_snorm_t>(*this);
}

void activation_layer_snorm_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
{
        odata = idata / (1 + idata.square()).sqrt();
}

void activation_layer_snorm_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
{
        idata = odata / (1 + idata.square()).cube().sqrt();
}

rlayer_t activation_layer_tanh_t::clone() const
{
        return std::make_unique<activation_layer_tanh_t>(*this);
}

void activation_layer_tanh_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
{
        odata = idata.tanh();
}

void activation_layer_tanh_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
{
        idata = odata * 4 / (idata.exp() + (-idata).exp()).square();
}

rlayer_t activation_layer_sigm_t::clone() const
{
        return std::make_unique<activation_layer_sigm_t>(*this);
}

void activation_layer_sigm_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
{
        odata = idata.exp() / (1 + idata.exp());
}

void activation_layer_sigm_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
{
        idata = odata * idata.exp() / (1 + idata.exp()).square();
}

rlayer_t activation_layer_splus_t::clone() const
{
        return std::make_unique<activation_layer_splus_t>(*this);
}

void activation_layer_splus_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
{
        odata = (1 + idata.exp()).log();
}

void activation_layer_splus_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
{
        idata = odata * idata.exp() / (1 + idata.exp());
}

rlayer_t activation_layer_unit_t::clone() const
{
        return std::make_unique<activation_layer_unit_t>(*this);
}

void activation_layer_unit_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
{
        odata = idata;
}

void activation_layer_unit_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
{
        idata = odata;
}

rlayer_t activation_layer_pwave_t::clone() const
{
        return std::make_unique<activation_layer_pwave_t>(*this);
}

void activation_layer_pwave_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
{
        odata = idata / (1 + idata.square());
}

void activation_layer_pwave_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
{
        idata = odata * (1 - idata.square()) / (1 + idata.square()).square();
}
