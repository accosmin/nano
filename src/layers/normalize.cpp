#include "normalize.h"
#include "math/numeric.h"

using namespace nano;

normalize_layer_t::normalize_layer_t(const string_t& parameters) :
        layer_t(parameters),
        m_idims({0, 0, 0}),
        m_odims({0, 0, 0})
{
}

void normalize_layer_t::configure(const tensor3d_dims_t& idims, const string_t& name)
{
        m_idims = idims;
        m_odims = idims;

        m_probe_output = probe_t{name, name + "(output)", 10 * isize()};
        m_probe_ginput = probe_t{name, name + "(ginput)", 10 * isize()};
        m_probe_gparam = probe_t{name, name + "(gparam)", 0};
}

void normalize_layer_t::output(tensor3d_const_map_t idata, tensor1d_const_map_t param, tensor3d_map_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());
        NANO_UNUSED1_RELEASE(param);

        m_probe_output.measure([&] ()
        {
                const auto sum = idata.array().sum();
                const auto sumsq = idata.array().square().sum();
                const auto count = static_cast<size_t>(idata.size());
                const auto avg = sum / count;
                const auto stdev = std::sqrt((sumsq - sum * sum / count) / count);
                odata.array() = (idata.array() - avg) / stdev;
        });
}

void normalize_layer_t::ginput(tensor3d_map_t idata, tensor1d_const_map_t param, tensor3d_const_map_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());
        NANO_UNUSED1_RELEASE(param);

        m_probe_ginput.measure([&] ()
        {
                idata.array() = odata.array();
        });
}

void normalize_layer_t::gparam(tensor3d_const_map_t idata, tensor1d_map_t param, tensor3d_const_map_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());
        NANO_UNUSED3_RELEASE(idata, param, odata);
}

rlayer_t normalize_layer_t::clone() const
{
        return std::make_unique<normalize_layer_t>(*this);
}
