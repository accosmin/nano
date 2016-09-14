#include "logger.h"
#include "pooling.hpp"
#include "pooling_soft.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
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
                        m_edata.vector(o) = (m_idata.vector(o) * m_wdata(o)).array().exp();

                        pooling::output(m_edata.matrix(o), m_odata.matrix(o), [] (const auto& ivec)
                        {
                                return ivec.sum();
                        });
                }

                const auto delta = std::log(scalar_t(9));
                m_odata.vector() = m_odata.vector().array().log() - delta;

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
                        const auto wdata = m_wdata(o);

                        pooling::ginput(m_idata.matrix(o), m_odata.matrix(o), m_edata.matrix(o), [=] (const auto ooo,
                                auto& i00, auto& i01, auto& i02,
                                auto& i10, auto& i11, auto& i12,
                                auto& i20, auto& i21, auto& i22,
                                const auto e00, const auto e01, const auto e02,
                                const auto e10, const auto e11, const auto e12,
                                const auto e20, const auto e21, const auto e22)
                        {
                                const auto sum = e00 + e01 + e02 + e10 + e11 + e12 + e20 + e21 + e22;
                                const auto wei = ooo * wdata / sum;

                                i00 += wei * e00; i01 += wei * e01; i02 += wei * e02;
                                i10 += wei * e10; i11 += wei * e11; i12 += wei * e12;
                                i20 += wei * e20; i21 += wei * e21; i22 += wei * e22;
                        });
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
                        auto& wdata = gwdata(o);
                        wdata = 0;

                        pooling::gparam(m_idata.matrix(o), m_odata.matrix(o), m_edata.matrix(o), [&] (const auto ooo,
                                const auto& ivec, const auto& evec)
                        {
                                wdata += ooo / evec.sum() * ivec.dot(evec);
                        });
                }
        }
}


