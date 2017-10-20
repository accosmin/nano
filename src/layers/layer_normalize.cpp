#include "layer_normalize.h"

using namespace nano;

template <typename tiarray, typename toarray>
static void onorm(const tiarray& iarray, toarray&& oarray)
{
        assert(std::isfinite(iarray.minCoeff()));
        assert(std::isfinite(iarray.maxCoeff()));

        const auto isum1 = iarray.sum();
        const auto isum2 = iarray.square().sum();
        const auto count = static_cast<scalar_t>(iarray.size());
        const auto imean = isum1 / count;
        const auto istdv = std::sqrt((isum2 - isum1 * isum1 / count) / count);

        oarray = (iarray - imean) / istdv;

        assert(std::isfinite(oarray.minCoeff()));
        assert(std::isfinite(oarray.maxCoeff()));
}

template <typename tiarray, typename toarray>
static void gnorm(tiarray&& iarray, const toarray& oarray)
{
        assert(std::isfinite(iarray.minCoeff()));
        assert(std::isfinite(iarray.maxCoeff()));
        assert(std::isfinite(oarray.minCoeff()));
        assert(std::isfinite(oarray.maxCoeff()));

        const auto isum1 = iarray.sum();
        const auto isum2 = iarray.square().sum();
        const auto count = static_cast<scalar_t>(iarray.size());
        const auto imean = isum1 / count;
        const auto istdv = std::sqrt((isum2 - isum1 * isum1 / count) / count);

        const auto osum1 = oarray.sum();
        const auto oisum = (oarray * (iarray - imean)).sum();

        iarray = oarray / (istdv) -
                 osum1 / (count * istdv) -
                 (iarray - imean) * oisum / (count * istdv * istdv * istdv);

        assert(std::isfinite(iarray.minCoeff()));
        assert(std::isfinite(iarray.maxCoeff()));
}

normalize_layer_t::normalize_layer_t(const string_t& params) :
        layer_t(to_params(params, "type", norm_type::plane)),
        m_xdims({0, 0, 0}),
        m_type(norm_type::plane)
{
}

rlayer_t normalize_layer_t::clone() const
{
        return std::make_unique<normalize_layer_t>(*this);
}

void normalize_layer_t::configure(const tensor3d_dims_t& idims, const string_t& name)
{
        m_xdims = idims;
        m_type = from_params<norm_type>(config(), "type");

        m_probe_output = probe_t{name, name + "(output)", 5 * isize()};
        m_probe_ginput = probe_t{name, name + "(ginput)", 12 * isize()};
        m_probe_gparam = probe_t{name, name + "(gparam)", 0};
}

void normalize_layer_t::output(const tensor4d_t& idata, const tensor1d_t& pdata, tensor4d_t& odata)
{
        assert(idata.dims() == odata.dims());
        assert(pdata.dims() == pdims());
        NANO_UNUSED1_RELEASE(pdata);

        const auto count = idata.size<0>();
        const auto imaps = idata.size<1>();
        m_probe_output.measure([&] ()
        {
                switch (m_type)
                {
                case norm_type::global:
                        for (auto x = 0; x < count; ++ x)
                        {
                                onorm(idata.array(x), odata.array(x));
                        }
                        break;
                case norm_type::plane:
                        for (auto x = 0; x < count; ++ x)
                        {
                                for (auto i = 0; i < imaps; ++ i)
                                {
                                        onorm(idata.array(x, i), odata.array(x, i));
                                }
                        }
                        break;
                }
        }, count);
}

void normalize_layer_t::ginput(tensor4d_t& idata, const tensor1d_t& pdata, const tensor4d_t& odata)
{
        assert(idata.dims() == odata.dims());
        assert(pdata.dims() == pdims());
        NANO_UNUSED1_RELEASE(pdata);

        const auto count = idata.size<0>();
        const auto imaps = idata.size<1>();
        m_probe_ginput.measure([&] ()
        {
                switch (m_type)
                {
                case norm_type::global:
                        for (auto x = 0; x < count; ++ x)
                        {
                                gnorm(idata.array(x), odata.array(x));
                        }
                        break;
                case norm_type::plane:
                        for (auto x = 0; x < count; ++ x)
                        {
                                for (auto i = 0; i < imaps; ++ i)
                                {
                                        gnorm(idata.array(x, i), odata.array(x, i));
                                }
                        }
                        break;
                }
        }, count);
}

void normalize_layer_t::gparam(const tensor4d_t& idata, tensor1d_t& pdata, const tensor4d_t& odata)
{
        assert(idata.dims() == odata.dims());
        assert(pdata.dims() == pdims());
        NANO_UNUSED3_RELEASE(idata, pdata, odata);
}
