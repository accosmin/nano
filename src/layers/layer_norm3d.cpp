#include "layer_norm3d.h"

using namespace nano;

json_reader_t& norm3d_layer_t::config(json_reader_t& reader)
{
        return reader.object("type", m_params.m_ntype);
}

json_writer_t& norm3d_layer_t::config(json_writer_t& writer) const
{
        return writer.object("type", m_params.m_ntype, "types", join(enum_values<norm_type>()));
}

rlayer_t norm3d_layer_t::clone() const
{
        return std::make_unique<norm3d_layer_t>(*this);
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
        NANO_UNUSED3_RELEASE(idata, pdata, odata);
}
