#include "math/random.h"
#include "layer_affine.h"
#include "tensor/numeric.h"

using namespace nano;

void affine_layer_t::from_json(const json_t& json)
{
        nano::from_json(json, "omaps", m_params.m_omaps, "orows", m_params.m_orows, "ocols", m_params.m_ocols);
}

void affine_layer_t::to_json(json_t& json) const
{
        nano::to_json(json, "omaps", m_params.m_omaps, "orows", m_params.m_orows, "ocols", m_params.m_ocols);
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

void affine_layer_t::random(vector_map_t pdata) const
{
        assert(pdata.size() == psize());

        const auto fanin = static_cast<scalar_t>(m_params.isize());
        const auto wmin = -std::sqrt(1 / (1 + fanin));
        const auto wmax = +std::sqrt(1 / (1 + fanin));

        const auto bmin = scalar_t(-0.1);
        const auto bmax = scalar_t(+0.1);

        nano::set_random(make_udist<scalar_t>(wmin, wmax), make_rng(), wdata(pdata));
        nano::set_random(make_udist<scalar_t>(bmin, bmax), make_rng(), bdata(pdata));
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
