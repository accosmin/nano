#include "layer_plus4d.h"

using namespace nano;

rlayer_t plus4d_layer_t::clone() const
{
        return std::make_unique<plus4d_layer_t>(*this);
}

bool plus4d_layer_t::resize(const tensor3d_dims_t& idims)
{
        if (idims.size() < 2)
        {
                return false;
        }

        for (const auto& idim : idims)
        {
                if (idim != idims[0])
                {
                        return false;
                }
        }

        m_idims = idims;
        m_odims = idims[0];

        m_fanin = static_cast<tensor_size_t>(idims.size());
        m_isize = m_fanin * nano::size(m_odims);
        return true;
}

void plus4d_layer_t::output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata)
{
        const auto count = odata.size<0>();
        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);

        assert(m_fanin > 0);
        assert(idata.size() == static_cast<size_t>(m_fanin));
        assert(idata[0].dims() == odata.dims());

        odata = idata[0];
        for (size_t i = 1; i < idata.size(); ++ i)
        {
                assert(idata[i].dims() == odata.dims());
                odata.array() += idata[i].array();
        }
}

void plus4d_layer_t::ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata)
{
        const auto count = odata.size<0>();
        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);

        assert(idata.size() == static_cast<size_t>(m_fanin));
        assert(idata[0].dims() == odata.dims());

        for (size_t i = 0; i < idata.size(); ++ i)
        {
                assert(idata[i].dims() == odata.dims());
                idata[i] = odata;
        }
}

void plus4d_layer_t::gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata)
{
        const auto count = odata.size<0>();
        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());
        NANO_UNUSED3_RELEASE(idata, pdata, odata);

        assert(m_fanin > 0);
        assert(idata.size() == static_cast<size_t>(m_fanin));
        assert(idata[0].dims() == odata.dims());
}
