#include "layer_tcat4d.h"

using namespace nano;

rlayer_t tcat4d_layer_t::clone() const
{
        return std::make_unique<tcat4d_layer_t>(*this);
}

bool tcat4d_layer_t::resize(const tensor3d_dims_t& idims)
{
        if (idims.size() < 2)
        {
                return false;
        }

        const auto irows = std::get<1>(idims[0]);
        const auto icols = std::get<2>(idims[0]);

        tensor_size_t imaps = 0;
        for (const auto& idim : idims)
        {
                if(     std::get<1>(idim) != irows ||
                        std::get<2>(idim) != icols)
                {
                        return false;
                }
                imaps += std::get<0>(idim);
        }

        m_odims = make_dims(imaps, irows, icols);
        return true;
}

void tcat4d_layer_t::output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata)
{
        const auto count = odata.size<0>();
        const auto omaps = odata.size<1>();
        const auto orows = odata.size<2>();
        const auto ocols = odata.size<3>();

        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);

        tensor_size_t imaps_offset = 0, odata_offset = 0;
        for (const auto& itensor : idata)
        {
                assert(itensor.size<0>() == count);
                assert(itensor.size<1>() + imaps_offset <= omaps);
                assert(itensor.size<2>() == orows);
                assert(itensor.size<3>() == ocols);

                const auto imaps = itensor.size<1>();
                const auto isize = imaps * orows * ocols;
                for (tensor_size_t x = 0; x < count; ++ x)
                {
                        odata.vector(x).segment(odata_offset, isize) = itensor.vector(x);
                }

                imaps_offset += imaps;
                odata_offset += isize;
        }

        assert(imaps_offset == omaps);
}

void tcat4d_layer_t::ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata)
{
        const auto count = odata.size<0>();
        const auto omaps = odata.size<1>();
        const auto orows = odata.size<2>();
        const auto ocols = odata.size<3>();

        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);

        tensor_size_t imaps_offset = 0, odata_offset = 0;
        for (const auto& itensor : idata)
        {
                assert(itensor.size<0>() == count);
                assert(itensor.size<1>() + imaps_offset <= omaps);
                assert(itensor.size<2>() == orows);
                assert(itensor.size<3>() == ocols);

                const auto imaps = itensor.size<1>();
                const auto isize = imaps * orows * ocols;
                for (tensor_size_t x = 0; x < count; ++ x)
                {
                        itensor.vector(x) = odata.vector(x).segment(odata_offset, isize);
                }

                imaps_offset += imaps;
                odata_offset += isize;
        }

        assert(imaps_offset == omaps);
}

void tcat4d_layer_t::gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata)
{
        const auto count = odata.size<0>();
        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());
        NANO_UNUSED3_RELEASE(idata, pdata, odata);
}
