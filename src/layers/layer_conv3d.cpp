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

bool conv3d_layer_t::resize(const tensor3d_dims_t& idims)
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

        m_kernel = conv4d_t{m_params};
        return true;
}

void conv3d_layer_t::output(tensor4d_cmap_t idata, vector_cmap_t pdata, tensor4d_map_t odata)
{
        m_kernel.output(idata, kdata(pdata), bdata(pdata), odata);
}

void conv3d_layer_t::ginput(tensor4d_map_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata)
{
        m_kernel.ginput(idata, kdata(pdata), bdata(pdata), odata);
}

void conv3d_layer_t::gparam(tensor4d_cmap_t idata, vector_map_t pdata, tensor4d_cmap_t odata)
{
        m_kernel.gparam(idata, kdata(pdata), bdata(pdata), odata);
}
