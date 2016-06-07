#include "pooling.hpp"
#include "pooling_soft.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "cortex/logger.h"
#include "tensor/numeric.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"

namespace nano
{
        pooling_soft_layer_t::pooling_soft_layer_t(const string_t& parameters) :
                layer_t(parameters)
        {
        }

        tensor_size_t pooling_soft_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = idims;
                const auto orows = irows / 2;
                const auto ocols = icols / 2;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_edata.resize(idims, irows, icols);
                m_wdata.resize(odims);

                return psize();
        }

        void pooling_soft_layer_t::zero_params()
        {
                tensor::set_zero(m_wdata);
        }

        void pooling_soft_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(nano::random_t<scalar_t>(min, max), m_wdata);
        }

        scalar_t* pooling_soft_layer_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_wdata);
        }

        const scalar_t* pooling_soft_layer_t::load_params(const scalar_t* params)
        {
                return tensor::from_array(params, m_wdata);
        }

        const tensor3d_t& pooling_soft_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() <= input.size<1>());
                assert(icols() <= input.size<2>());

                m_idata = input;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto idata = m_idata.matrix(o);
                        const auto wdata = m_wdata(o);

                        auto edata = m_edata.matrix(o);
                        edata = (idata.array() * wdata).exp();

                        pooling::output(edata, m_odata.matrix(o), [&] (
                                const auto i00, const auto i01, const auto i02,
                                const auto i10, const auto i11, const auto i12,
                                const auto i20, const auto i21, const auto i22)
                        {
                                return std::log((i00 + i01 + i02 + i10 + i11 + i12 + i20 + i21 + i22) / 9);
                        });
                }

                return m_odata;
        }

        const tensor3d_t& pooling_soft_layer_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto odata = m_odata.matrix(o);
                        const auto edata = m_edata.matrix(o);
                        const auto wdata = m_wdata(o);

                        auto idata = m_idata.matrix(o);
                        idata.setZero();

                        for (tensor_size_t r = 1; r < idata.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < idata.cols(); c += 2)
                                {
                                        const auto c0 = c - 1, c1 = c, c2 = std::min(c + 1, idata.cols() - 1);
                                        const auto r0 = r - 1, r1 = r, r2 = std::min(r + 1, idata.rows() - 1);

                                        const auto sum =
                                        edata(r0, c0) + edata(r0, c1) + edata(r0, c2) +
                                        edata(r1, c0) + edata(r1, c1) + edata(r1, c2) +
                                        edata(r2, c0) + edata(r2, c1) + edata(r2, c2);

                                        const auto wei = odata(r / 2, c / 2) / sum * wdata;

                                        idata(r0, c0) += wei * edata(r0, c0);
                                        idata(r0, c1) += wei * edata(r0, c1);
                                        idata(r0, c2) += wei * edata(r0, c2);

                                        idata(r1, c0) += wei * edata(r1, c0);
                                        idata(r1, c1) += wei * edata(r1, c1);
                                        idata(r1, c2) += wei * edata(r1, c2);

                                        idata(r2, c0) += wei * edata(r2, c0);
                                        idata(r2, c1) += wei * edata(r2, c1);
                                        idata(r2, c2) += wei * edata(r2, c2);
                                }
                        }
                }

                return m_idata;
        }

        void pooling_soft_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                auto gwdata = tensor::map_vector(gradient, odims());

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto idata = m_idata.matrix(o);
                        const auto edata = m_edata.matrix(o);
                        const auto odata = m_odata.matrix(o);

                        auto& wdata = gwdata(o);
                        wdata = 0;

                        for (tensor_size_t r = 1; r < idata.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < idata.cols(); c += 2)
                                {
                                        const auto c0 = c - 1, c1 = c, c2 = std::min(c + 1, idata.cols() - 1);
                                        const auto r0 = r - 1, r1 = r, r2 = std::min(r + 1, idata.rows() - 1);

                                        const auto sum =
                                        edata(r0, c0) + edata(r0, c1) + edata(r0, c2) +
                                        edata(r1, c0) + edata(r1, c1) + edata(r1, c2) +
                                        edata(r2, c0) + edata(r2, c1) + edata(r2, c2);

                                        wdata += odata(r / 2, c / 2) / sum * (
                                        edata(r0, c0) * idata(r0, c0) +
                                        edata(r0, c1) * idata(r0, c1) +
                                        edata(r0, c2) * idata(r0, c2) +

                                        edata(r1, c0) * idata(r1, c0) +
                                        edata(r1, c1) * idata(r1, c1) +
                                        edata(r1, c2) * idata(r1, c2) +

                                        edata(r2, c0) * idata(r2, c0) +
                                        edata(r2, c1) * idata(r2, c1) +
                                        edata(r2, c2) * idata(r2, c2));
                                }
                        }
                }
        }
}


