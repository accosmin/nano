#include "math/random.h"
#include "layer_conv3d.h"
#include "tensor/numeric.h"

using namespace nano;

void conv3d_layer_t::from_json(const json_t& json)
{
        nano::from_json(json, "omaps", m_params.m_omaps, "krows", m_params.m_krows, "kcols", m_params.m_kcols,
                "kconn", m_params.m_kconn, "kdrow", m_params.m_kdrow, "kdcol", m_params.m_kdcol);
}

void conv3d_layer_t::to_json(json_t& json) const
{
        nano::to_json(json, "omaps", m_params.m_omaps, "krows", m_params.m_krows, "kcols", m_params.m_kcols,
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

void conv3d_layer_t::random(vector_map_t pdata) const
{
        assert(pdata.size() == psize());

        const auto fanin = static_cast<scalar_t>(imaps() * m_params.krows() * m_params.kcols()) / static_cast<scalar_t>(kconn());
        const auto kmin = -std::sqrt(1 / (1 + fanin));
        const auto kmax = +std::sqrt(1 / (1 + fanin));

        const auto bmin = scalar_t(-0.1);
        const auto bmax = scalar_t(+0.1);

        nano::set_random(make_udist<scalar_t>(kmin, kmax), make_rng(), kdata(pdata));
        nano::set_random(make_udist<scalar_t>(bmin, bmax), make_rng(), bdata(pdata));
}

void conv3d_layer_t::output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata)
{
        assert(idata.size() == 1);
        m_kernel.output(idata[0], kdata(pdata), bdata(pdata), odata);
}

void conv3d_layer_t::ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata)
{
        assert(idata.size() == 1);
        m_kernel.ginput(idata[0], kdata(pdata), bdata(pdata), odata);
}

void conv3d_layer_t::gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata)
{
        assert(idata.size() == 1);
        m_kernel.gparam(idata[0], kdata(pdata), bdata(pdata), odata);
}
