#include "layer_normalize.h"

using namespace nano;

json_reader_t& normalize_layer_t::config(json_reader_t& reader)
{
        return reader.object("type", m_params.m_ntype);
}

json_writer_t& normalize_layer_t::config(json_writer_t& writer) const
{
        return writer.object("type", m_params.m_ntype, "types", join(enum_values<norm_type>()));
}

rlayer_t normalize_layer_t::clone() const
{
        return std::make_unique<normalize_layer_t>(*this);
}

bool normalize_layer_t::resize(const tensor3d_dims_t& idims, const string_t& name)
{
        m_params.m_xmaps = std::get<0>(idims);
        m_params.m_xrows = std::get<1>(idims);
        m_params.m_xcols = std::get<2>(idims);

        if (!m_params.valid())
        {
                return false;
        }

        m_kernel = norm4d_t{m_params};

        m_probe_output = probe_t{name, name + "(output)", m_params.flops_output()};
        m_probe_ginput = probe_t{name, name + "(ginput)", m_params.flops_ginput()};
        m_probe_gparam = probe_t{name, name + "(gparam)", m_params.flops_gparam()};
        return true;
}

void normalize_layer_t::output(const tensor4d_cmap_t& idata, const vector_cmap_t& pdata, tensor4d_map_t&& odata)
{
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);

        const auto count = idata.size<0>();
        m_probe_output.measure([&] () { m_kernel.output(idata, odata); }, count);
}

void normalize_layer_t::ginput(tensor4d_map_t&& idata, const vector_cmap_t& pdata, const tensor4d_cmap_t& odata)
{
        assert(pdata.size() == psize());
        NANO_UNUSED1_RELEASE(pdata);

        const auto count = idata.size<0>();
        m_probe_ginput.measure([&] () { m_kernel.ginput(idata, odata); }, count);
}

void normalize_layer_t::gparam(const tensor4d_cmap_t& idata, vector_map_t&& pdata, const tensor4d_cmap_t& odata)
{
        assert(idata.dims() == odata.dims());
        assert(pdata.size() == psize());
        NANO_UNUSED3_RELEASE(idata, pdata, odata);
}
