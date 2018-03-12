#include "layer_norm3d.h"

using namespace nano;

void norm3d_layer_t::from_json(const json_t& json)
{
        nano::from_json(json, "norm", m_params.m_ntype);
}

void norm3d_layer_t::to_json(json_t& json) const
{
        nano::to_json(json, "norm", m_params.m_ntype, "norm", join(enum_values<norm_type>()));
}

rlayer_t norm3d_layer_t::clone() const
{
        return std::make_unique<norm3d_layer_t>(*this);
}

void norm3d_layer_t::random(vector_map_t pdata) const
{
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);
}

bool norm3d_layer_t::resize(const tensor3d_dims_t& idims)
{
        if (idims.size() != 1)
        {
                return false;
        }

        m_params.m_xmaps = std::get<0>(idims[0]);
        m_params.m_xrows = std::get<1>(idims[0]);
        m_params.m_xcols = std::get<2>(idims[0]);
        if (!m_params.valid())
        {
                return false;
        }

        m_kernel = norm4d_t{m_params};
        return true;
}

void norm3d_layer_t::output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata)
{
        assert(idata.size() == 1);
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);

        m_kernel.output(idata[0], odata);
}

void norm3d_layer_t::ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata)
{
        assert(idata.size() == 1);
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);

        m_kernel.ginput(idata[0], odata);
}

void norm3d_layer_t::gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata)
{
        assert(idata.size() == 1);
        assert(pdata.size() == psize());
        assert(idata[0].dims() == odata.dims());
        NANO_UNUSED3_RELEASE(idata, pdata, odata);
}
