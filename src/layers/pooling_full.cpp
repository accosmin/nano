#include "pooling.hpp"
#include "pooling_full.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "logger.h"
#include "tensor/numeric.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"

namespace nano
{
        pooling_full_layer_t::pooling_full_layer_t(const string_t& parameters) :
                layer_t(parameters)
        {
        }

        tensor_size_t pooling_full_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = idims;
                const auto orows = irows / 2;
                const auto ocols = icols / 2;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_wdata.resize(odims, 3, 3);

                return psize();
        }

        void pooling_full_layer_t::zero_params()
        {
                tensor::set_zero(m_wdata);
        }

        void pooling_full_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(nano::random_t<scalar_t>(min, max), m_wdata);
        }

        scalar_t* pooling_full_layer_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_wdata);
        }

        const scalar_t* pooling_full_layer_t::load_params(const scalar_t* params)
        {
                return tensor::from_array(params, m_wdata);
        }

        const tensor3d_t& pooling_full_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() <= input.size<1>());
                assert(icols() <= input.size<2>());

                m_idata = input;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto wdata = m_wdata.matrix(o);
                        const auto wnorm = scalar_t(1) / std::sqrt(wdata.array().square().sum() + 1);

                        pooling::output(m_idata.matrix(o), m_odata.matrix(o), [&] (
                                const auto i00, const auto i01, const auto i02,
                                const auto i10, const auto i11, const auto i12,
                                const auto i20, const auto i21, const auto i22)
                        {
                                const auto sum =
                                        i00 * wdata(0, 0) + i01 * wdata(0, 1) + i02 * wdata(0, 2) +
                                        i10 * wdata(1, 0) + i11 * wdata(1, 1) + i12 * wdata(1, 2) +
                                        i20 * wdata(2, 0) + i21 * wdata(2, 1) + i22 * wdata(2, 2);
                                return wnorm * sum;
                        });
                }

                return m_odata;
        }

        const tensor3d_t& pooling_full_layer_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto wdata = m_wdata.matrix(o);
                        const auto wnorm = scalar_t(1) / std::sqrt(wdata.array().square().sum() + 1);

                        pooling::ginput(m_idata.matrix(o), m_odata.matrix(o), [&] (const auto ooo_,
                                auto& i00, auto& i01, auto& i02,
                                auto& i10, auto& i11, auto& i12,
                                auto& i20, auto& i21, auto& i22)
                        {
                                const auto ooo = wnorm * ooo_;
                                i00 += ooo * wdata(0, 0); i01 += ooo * wdata(0, 1); i02 += ooo * wdata(0, 2);
                                i10 += ooo * wdata(1, 0); i11 += ooo * wdata(1, 1); i12 += ooo * wdata(1, 2);
                                i20 += ooo * wdata(2, 0); i21 += ooo * wdata(2, 1); i22 += ooo * wdata(2, 2);
                        });
                }

                return m_idata;
        }

        void pooling_full_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                auto gwdata = tensor::map_tensor(gradient, odims(), 3, 3);

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        matrix_t gdata = matrix_t::Zero(3, 3);

                        pooling::gparam(m_idata.matrix(o), m_odata.matrix(o), [&] (
                                const auto g00, const auto g01, const auto g02,
                                const auto g10, const auto g11, const auto g12,
                                const auto g20, const auto g21, const auto g22)
                        {
                                gdata(0, 0) += g00; gdata(0, 1) += g01; gdata(0, 2) += g02;
                                gdata(1, 0) += g10; gdata(1, 1) += g11; gdata(1, 2) += g12;
                                gdata(2, 0) += g20; gdata(2, 1) += g21; gdata(2, 2) += g22;
                        });

                        const auto wdata = m_wdata.matrix(o);
                        const auto ww = (wdata.array() * wdata.array()).sum();
                        const auto wg = (gdata.array() * wdata.array()).sum();
                        const auto wnorm = scalar_t(1) / ((ww + 1) * std::sqrt(ww + 1));
                        gwdata.matrix(o) = wnorm * (gdata.array() * (ww + 1) - wdata.array() * wg);
                }
        }
}


