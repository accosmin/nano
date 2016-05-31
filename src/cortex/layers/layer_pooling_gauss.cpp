#include "pooling.hpp"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "cortex/logger.h"
#include "math/numeric.hpp"
#include "tensor/numeric.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"
#include "layer_pooling_gauss.h"

namespace nano
{
        template <typename tmatrix>
        static matrix_t make_gauss(const tmatrix& gdata)
        {
                const auto meanx = gdata(0, 0);
                const auto meany = gdata(0, 1);
                const auto precx = gdata(1, 0);
                const auto precy = gdata(1, 1);

                matrix_t wei(3, 3);
                for (int y = -1; y <= 1; ++ y)
                {
                        for (int x = -1; x <= 1; ++ x)
                        {
                                wei(y + 1, x + 1) =
                                std::exp(- precx / 2 * square(x - meanx) - precy / 2 * square(y - meany));
                        }
                }

                return wei;
        }

        template <typename tmatrix>
        static matrix_t make_gauss_weights(const tmatrix& gdata)
        {
                const auto wei = make_gauss(gdata);
                const auto div = 1 / wei.sum();
                return wei.array().sqrt() * std::sqrt(div);
        }

        template <typename tmatrix>
        static matrix_t make_gauss_gradient(const tmatrix& gdata)
        {
                const auto wei = make_gauss(gdata);
                const auto sum = wei.sum();

                const auto meanx = gdata(0, 0);
                const auto meany = gdata(0, 1);
                const auto precx = gdata(1, 0);
                const auto precy = gdata(1, 1);

                matrix_t agrad(9, 4);
                for (int y = -1, i = 0; y <= 1; ++ y)
                {
                        for (int x = -1; x <= 1; ++ x, ++ i)
                        {
                                agrad(i, 0) = wei(y + 1, x + 1) * precx * (x - meanx);
                                agrad(i, 1) = wei(y + 1, x + 1) * precy * (y - meany);
                                agrad(i, 2) = wei(y + 1, x + 1) * (- square(x - meanx) / 2);
                                agrad(i, 3) = wei(y + 1, x + 1) * (- square(y - meany) / 2);
                        }
                }

                const auto sum_agrad0 = agrad.col(0).sum();
                const auto sum_agrad1 = agrad.col(1).sum();
                const auto sum_agrad2 = agrad.col(2).sum();
                const auto sum_agrad3 = agrad.col(3).sum();

                matrix_t grad(9, 4);
                for (int i = 0; i < 9; ++ i)
                {
                        const auto div = 1 / (2 * sum * std::sqrt(sum * wei(i)));
                        grad(i, 0) = (sum * agrad(i, 0) - wei(i) * sum_agrad0) * div;
                        grad(i, 1) = (sum * agrad(i, 1) - wei(i) * sum_agrad1) * div;
                        grad(i, 2) = (sum * agrad(i, 2) - wei(i) * sum_agrad2) * div;
                        grad(i, 3) = (sum * agrad(i, 3) - wei(i) * sum_agrad3) * div;
                }

                return grad;
        }

        pooling_gauss_layer_t::pooling_gauss_layer_t(const string_t& parameters) :
                layer_t(parameters)
        {
        }

        tensor_size_t pooling_gauss_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = idims;
                const auto orows = irows / 2;
                const auto ocols = icols / 2;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_gdata.resize(odims, 2, 2);

                return psize();
        }

        void pooling_gauss_layer_t::zero_params()
        {
                tensor::set_zero(m_gdata);
        }

        void pooling_gauss_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(nano::random_t<scalar_t>(min, max), m_gdata);
        }

        scalar_t* pooling_gauss_layer_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_gdata);
        }

        const scalar_t* pooling_gauss_layer_t::load_params(const scalar_t* params)
        {
                return tensor::from_array(params, m_gdata);
        }

        const tensor3d_t& pooling_gauss_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() <= input.size<1>());
                assert(icols() <= input.size<2>());

                m_idata = input;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto gauss = make_gauss_weights(m_gdata.matrix(o));

                        pooling::output(m_idata.matrix(o), m_odata.matrix(o), [&] (
                                const auto i00, const auto i01, const auto i02,
                                const auto i10, const auto i11, const auto i12,
                                const auto i20, const auto i21, const auto i22)
                        {
                                return  i00 * gauss(0, 0) + i01 * gauss(0, 1) + i02 * gauss(0, 2) +
                                        i10 * gauss(1, 0) + i11 * gauss(1, 1) + i12 * gauss(1, 2) +
                                        i20 * gauss(2, 0) + i21 * gauss(2, 1) + i22 * gauss(2, 2);
                        });
                }

                return m_odata;
        }

        const tensor3d_t& pooling_gauss_layer_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto gauss = make_gauss_weights(m_gdata.matrix(o));

                        pooling::ginput(m_idata.matrix(o), m_odata.matrix(o), [&] (const auto ooo,
                                auto& i00, auto& i01, auto& i02,
                                auto& i10, auto& i11, auto& i12,
                                auto& i20, auto& i21, auto& i22)
                        {
                                i00 += ooo * gauss(0, 0); i01 += ooo * gauss(0, 1); i02 += ooo * gauss(0, 2);
                                i10 += ooo * gauss(1, 0); i11 += ooo * gauss(1, 1); i12 += ooo * gauss(1, 2);
                                i20 += ooo * gauss(2, 0); i21 += ooo * gauss(2, 1); i22 += ooo * gauss(2, 2);
                        });
                }

                return m_idata;
        }

        void pooling_gauss_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                auto ggdata = tensor::map_tensor(gradient, odims(), 2, 2);

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto gauss = make_gauss_gradient(m_gdata.matrix(o));

                        auto gdata = ggdata.vector(o);
                        gdata.setZero();

                        vector_t gg(9);

                        pooling::gparam(m_idata.matrix(o), m_odata.matrix(o), [&] (
                                const auto g00, const auto g01, const auto g02,
                                const auto g10, const auto g11, const auto g12,
                                const auto g20, const auto g21, const auto g22)
                        {
                                gg(0) = g00; gg(1) = g01; gg(2) = g02;
                                gg(3) = g10; gg(4) = g11; gg(5) = g12;
                                gg(6) = g20; gg(7) = g21; gg(8) = g22;

                                gdata += gauss.transpose() * gg;
                        });
                }
        }
}


