#include "layer_affine.h"
#include "math/numeric.h"

using namespace nano;

affine_layer_t::affine_layer_t(const string_t& params) :
        layer_t(to_params(params, "dims", "10[1,4096]")),
        m_idims({0, 0, 0}),
        m_odims({0, 0, 0}),
        m_psize(0)
{
}

rlayer_t affine_layer_t::clone() const
{
        return std::make_unique<affine_layer_t>(*this);
}

void affine_layer_t::configure(const tensor3d_dims_t& idims, const string_t& name)
{
        m_idims = idims;
        m_odims = {nano::clamp(nano::from_params<tensor_size_t>(config(), "dims"), 1, 4096), 1, 1};
        m_psize = isize() * osize() + osize();

        m_probe_output = probe_t{name, name + "(output)", isize() * osize() + osize()};
        m_probe_ginput = probe_t{name, name + "(ginput)", isize() * osize()};
        m_probe_gparam = probe_t{name, name + "(gparam)", isize() * osize()};
}

tensor_size_t affine_layer_t::fanin() const
{
        return isize();
}

void affine_layer_t::output(tensor3d_const_map_t idata, tensor1d_const_map_t param, tensor3d_map_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());

        m_probe_output.measure([&] ()
        {
                odata.vector() = wdata(param) * idata.vector() + bdata(param);
        });
}

void affine_layer_t::ginput(tensor3d_map_t idata, tensor1d_const_map_t param, tensor3d_const_map_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());

        m_probe_ginput.measure([&] ()
        {
                idata.vector() = wdata(param).transpose() * odata.vector();
        });
}

void affine_layer_t::gparam(tensor3d_const_map_t idata, tensor1d_map_t param, tensor3d_const_map_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());

        m_probe_gparam.measure([&] ()
        {
                bdata(param) = odata.vector();
                wdata(param) = odata.vector() * idata.vector().transpose();
        });
}
