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

bool affine_layer_t::resize(const tensor3d_dims_t& idims)
{
        if (idims.size() != 1)
        {
                return false;
        }

        m_params.m_imaps = std::get<0>(idims[0]);
        m_params.m_irows = std::get<1>(idims[0]);
        m_params.m_icols = std::get<2>(idims[0]);
        if (!m_params.valid())
        {
                return false;
        }

        m_kernel = affine4d_t{m_params};
        return true;
}

void affine_layer_t::output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata)
{
        assert(idata.size() == 1);
        m_kernel.output(idata[0], wdata(pdata), bdata(pdata), odata);
}

void affine_layer_t::ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata)
{
        assert(idata.size() == 1);
        m_kernel.ginput(idata[0], wdata(pdata), bdata(pdata), odata);
}

void affine_layer_t::gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata)
{
        assert(idata.size() == 1);
        m_kernel.gparam(idata[0], wdata(pdata), bdata(pdata), odata);
}
