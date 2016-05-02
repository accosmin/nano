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
                                wei(y + 1, x + 1) = std::exp(
                                        - precx / 2 * square(x - meanx)
                                        - precy / 2 * square(y - meany));
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
                const auto grad = (sum - wei.array()) / (2 * sum * (sum * wei.array()).sqrt());

                const auto meanx = gdata(0, 0);
                const auto meany = gdata(0, 1);
                const auto precx = gdata(1, 0);
                const auto precy = gdata(1, 1);

                matrix_t ret(9, 4);
                for (int y = -1, i = 0; y <= 1; ++ y)
                {
                        for (int x = -1; x <= 1; ++ x, ++ i)
                        {
                                ret(i, 0) = grad(y + 1, x + 1) * wei(y + 1, x + 1) * precx * (x - meanx);
                                ret(i, 1) = grad(y + 1, x + 1) * wei(y + 1, x + 1) * precy * (y - meany);
                                ret(i, 2) = grad(y + 1, x + 1) * wei(y + 1, x + 1) * (- square(x - meanx) / 2);
                                ret(i, 3) = grad(y + 1, x + 1) * wei(y + 1, x + 1) * (- square(y - meany) / 2);
                        }
                }

                return ret;
        }

        const tensor3d_t& pooling_gauss_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() <= input.size<1>());
                assert(icols() <= input.size<2>());

                m_idata = input;

                matrix_t gaussa(3, 3);
                matrix_t gaussw(3, 3);

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto idata = m_idata.matrix(o);
                        const auto gdata = m_gdata.matrix(o);
                        const auto gauss = make_gauss_weights(gdata);

                        auto odata = m_odata.matrix(o);

                        for (tensor_size_t r = 1; r < idata.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < idata.cols(); c += 2)
                                {
                                        const auto c0 = c - 1, c1 = c, c2 = std::min(c + 1, idata.cols() - 1);
                                        const auto r0 = r - 1, r1 = r, r2 = std::min(r + 1, idata.rows() - 1);

                                        odata(r / 2, c / 2) =
                                        idata(r0, c0) * gauss(0, 0) +
                                        idata(r0, c1) * gauss(0, 1) +
                                        idata(r0, c2) * gauss(0, 2) +
                                        idata(r1, c0) * gauss(1, 0) +
                                        idata(r1, c1) * gauss(1, 1) +
                                        idata(r1, c2) * gauss(1, 2) +
                                        idata(r2, c0) * gauss(2, 0) +
                                        idata(r2, c1) * gauss(2, 1) +
                                        idata(r2, c2) * gauss(2, 2);
                                }
                        }
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
                        const auto odata = m_odata.matrix(o);
                        const auto gdata = m_gdata.matrix(o);
                        const auto gauss = make_gauss_weights(gdata);

                        auto idata = m_idata.matrix(o);
                        idata.setZero();

                        for (tensor_size_t r = 1; r < idata.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < idata.cols(); c += 2)
                                {
                                        const auto c0 = c - 1, c1 = c, c2 = std::min(c + 1, idata.cols() - 1);
                                        const auto r0 = r - 1, r1 = r, r2 = std::min(r + 1, idata.rows() - 1);

                                        idata(r0, c0) += odata(r / 2, c / 2) * gauss(0, 0);
                                        idata(r0, c1) += odata(r / 2, c / 2) * gauss(0, 1);
                                        idata(r0, c2) += odata(r / 2, c / 2) * gauss(0, 2);

                                        idata(r1, c0) += odata(r / 2, c / 2) * gauss(1, 0);
                                        idata(r1, c1) += odata(r / 2, c / 2) * gauss(1, 1);
                                        idata(r1, c2) += odata(r / 2, c / 2) * gauss(1, 2);

                                        idata(r2, c0) += odata(r / 2, c / 2) * gauss(2, 0);
                                        idata(r2, c1) += odata(r / 2, c / 2) * gauss(2, 1);
                                        idata(r2, c2) += odata(r / 2, c / 2) * gauss(2, 2);
                                }
                        }
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
                        const auto idata = m_idata.matrix(o);
                        const auto odata = m_odata.matrix(o);
                        const auto gauss = make_gauss_gradient(m_gdata.matrix(o));

                        vector_t ugrad(9);
                        vector_t gdata(4); gdata.setZero();

                        for (tensor_size_t r = 1; r < idata.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < idata.cols(); c += 2)
                                {
                                        const auto c0 = c - 1, c1 = c, c2 = std::min(c + 1, idata.cols() - 1);
                                        const auto r0 = r - 1, r1 = r, r2 = std::min(r + 1, idata.rows() - 1);

                                        ugrad(0) = odata(r / 2, c / 2) * idata(r0, c0);
                                        ugrad(1) = odata(r / 2, c / 2) * idata(r0, c1);
                                        ugrad(2) = odata(r / 2, c / 2) * idata(r0, c2);

                                        ugrad(3) = odata(r / 2, c / 2) * idata(r1, c0);
                                        ugrad(4) = odata(r / 2, c / 2) * idata(r1, c1);
                                        ugrad(5) = odata(r / 2, c / 2) * idata(r1, c2);

                                        ugrad(6) = odata(r / 2, c / 2) * idata(r2, c0);
                                        ugrad(7) = odata(r / 2, c / 2) * idata(r2, c1);
                                        ugrad(8) = odata(r / 2, c / 2) * idata(r2, c2);

                                        gdata += gauss.transpose() * ugrad;
                                }
                        }

                        ggdata.vector(o) = gdata.array();
                }
        }
}


