#include "layer_plus4d.h"

using namespace nano;

rlayer_t plus4d_layer_t::clone() const
{
        return std::make_unique<plus4d_layer_t>(*this);
}

bool plus4d_layer_t::resize(const tensor3d_dims_t&, const string_t&)
{
        // needs at least two inputs to sum together!
        return false;
}

bool plus4d_layer_t::resize(const std::vector<tensor3d_dims_t>& idims, const string_t& name)
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

        // output dimensions: a single input
        m_odims = idims[0];

        // input dimensions: all the inputs concatenated
        const auto count = static_cast<tensor_size_t>(idims.size());
        m_idims = make_dims(count * std::get<0>(m_odims), std::get<1>(m_odims), std::get<2>(m_odims));

        m_probe_output = probe_t{name, name + "(output)", nano::size(m_idims)};
        m_probe_ginput = probe_t{name, name + "(ginput)", nano::size(m_idims)};
        m_probe_gparam = probe_t{name, name + "(gparam)", 0};
        return true;
}

void plus4d_layer_t::output(const tensor4d_cmap_t& idata, const vector_cmap_t& pdata, tensor4d_map_t&& odata)
{
        const auto count = idata.size<0>();
        assert(idata.dims() == cat_dims(count, idims()));
        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());

        m_probe_output.measure([&] ()
        {
                odata.zero();
                for (auto i = 0; i < count; ++ i)
                {
                        odata.vector() += map_vector(idata.data() + i * odata.size(), odata.size());
                }
        }, count);
}

void plus4d_layer_t::ginput(tensor4d_map_t&& idata, const vector_cmap_t& pdata, const tensor4d_cmap_t& odata)
{
        const auto count = idata.size<0>();
        assert(idata.dims() == cat_dims(count, idims()));
        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());

        m_probe_output.measure([&] ()
        {
                for (auto i = 0; i < count; ++ i)
                {
                        map_vector(idata.data() + i * odata.size(), odata.size()) = odata.vector();
                }
        }, count);
}

void plus4d_layer_t::gparam(const tensor4d_cmap_t& idata, vector_map_t&& pdata, const tensor4d_cmap_t& odata)
{
        const auto count = idata.size<0>();
        assert(idata.dims() == cat_dims(count, idims()));
        assert(odata.dims() == cat_dims(count, odims()));
        assert(pdata.size() == psize());
        NANO_UNUSED3_RELEASE(count, pdata, odata);
}
