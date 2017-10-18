#include "layer_normalize.h"

using namespace nano;

template <typename tidata, typename todata>
static void onorm(const tidata& idata, todata&& odata)
{
        const auto isum1 = idata.array().sum();
        const auto isum2 = idata.array().square().sum();
        const auto count = static_cast<scalar_t>(idata.size());
        const auto imean = isum1 / count;
        const auto istdv = std::sqrt((isum2 - isum1 * isum1 / count) / count);

        odata.array() = (idata.array() - imean) / istdv;
}

template <typename tidata, typename todata>
static void gnorm(tidata&& idata, const todata& odata)
{
        const auto isum1 = idata.array().sum();
        const auto isum2 = idata.array().square().sum();
        const auto count = static_cast<scalar_t>(idata.size());
        const auto imean = isum1 / count;
        const auto istdv = std::sqrt((isum2 - isum1 * isum1 / count) / count);

        const auto osum1 = odata.array().sum();
        const auto oisum = (odata.array() * (idata.array() - imean)).sum();

        idata.array() =
                odata.array() / (istdv) -
                osum1 / (count * istdv) -
                (idata.array() - imean) * oisum / (count * istdv * istdv * istdv);
}

normalize_layer_t::normalize_layer_t(const string_t& params) :
        layer_t(to_params(params, "type", norm_type::plane)),
        m_idims({0, 0, 0}),
        m_odims({0, 0, 0}),
        m_type(norm_type::plane)
{
}

void normalize_layer_t::configure(const tensor3d_dims_t& idims, const string_t& name)
{
        m_idims = idims;
        m_odims = idims;
        m_type = from_params<norm_type>(config(), "type");

        m_probe_output = probe_t{name, name + "(output)", 5 * isize()};
        m_probe_ginput = probe_t{name, name + "(ginput)", 12 * isize()};
        m_probe_gparam = probe_t{name, name + "(gparam)", 0};
}

void normalize_layer_t::output(tensor3d_cmap_t idata, tensor1d_cmap_t param, tensor3d_map_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());
        NANO_UNUSED1_RELEASE(param);

        m_probe_output.measure([&] ()
        {
                switch (m_type)
                {
                case norm_type::global:
                        onorm(idata, odata);
                        break;
                case norm_type::plane:
                        for (tensor_size_t i = 0; i < std::get<0>(m_idims); ++ i)
                        {
                                onorm(idata.matrix(i), odata.matrix(i));
                        }
                        break;
                }
        });
}

void normalize_layer_t::ginput(tensor3d_map_t idata, tensor1d_cmap_t param, tensor3d_cmap_t odata)
{
        assert(idata.dims() == idims());
        assert(param.size() == psize());
        assert(odata.dims() == odims());
        NANO_UNUSED1_RELEASE(param);

        m_probe_ginput.measure([&] ()
        {
                switch (m_type)
                {
                case norm_type::global:
                        gnorm(idata, odata);
                        break;
                case norm_type::plane:
                        for (tensor_size_t i = 0; i < std::get<0>(m_idims); ++ i)
                        {
                                gnorm(idata.matrix(i), odata.matrix(i));
                        }
                        break;
                }
        });
}

void normalize_layer_t::gparam(tensor3d_cmap_t idata, tensor1d_map_t param, tensor3d_cmap_t odata)
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
