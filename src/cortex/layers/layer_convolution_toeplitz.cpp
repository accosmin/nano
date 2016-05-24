#include "math/clamp.hpp"
#include "math/random.hpp"
#include "cortex/logger.h"
#include "tensor/numeric.hpp"
#include "text/to_string.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"
#include "layer_convolution_toeplitz.h"

namespace nano
{
        template <typename timatrix, typename tsize, typename tomatrix>
        static void make_conv(const timatrix& imat, const tsize krows, const tsize kcols, tomatrix& omat)
        {
                const tsize irows = imat.rows();
                const tsize icols = imat.cols();
                const tsize orows = irows - krows + 1;
                const tsize ocols = icols - kcols + 1;

                assert(omat.rows() == krows * kcols);
                assert(omat.cols() == orows * ocols);

                for (tsize kr = 0; kr < krows; ++ kr)
                {
                        for (tsize kc = 0; kc < kcols; ++ kc)
                        {
                                for (tsize r = 0; r < orows; ++ r)
                                {
                                        for (tsize c = 0; c < ocols; ++ c)
                                        {
                                                omat(kr * kcols + kc, r * ocols + c) = imat(r + kr, c + kc);
                                        }
                                }
                        }
                }
        }

        template <typename tomatrix, typename tsize, typename timatrix>
        static void make_corr(const tomatrix& omat, const tsize krows, const tsize kcols, timatrix& imat)
        {
                const tsize orows = omat.rows();
                const tsize ocols = omat.cols();
                const tsize irows = orows + krows - 1;
                const tsize icols = ocols + kcols - 1;

                NANO_UNUSED1_RELEASE(irows);

                assert(imat.rows() == krows * kcols);
                assert(imat.cols() == irows * icols);

                imat.setZero();
                for (tsize kr = 0; kr < krows; ++ kr)
                {
                        for (tsize kc = 0; kc < kcols; ++ kc)
                        {
                                for (tsize r = 0; r < orows; ++ r)
                                {
                                        for (tsize c = 0; c < ocols; ++ c)
                                        {
                                                imat(kr * kcols + kc, (r + kr) * icols + c + kc) += omat(r, c);
                                        }
                                }
                        }
                }
        }

        conv_layer_toeplitz_t::conv_layer_toeplitz_t(const string_t& parameters) :
                layer_t(parameters),
                m_kconn(1)
        {
        }

        tensor_size_t conv_layer_toeplitz_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = clamp(from_params<tensor_size_t>(configuration(), "dims", 16), 1, 256);
                const auto krows = clamp(from_params<tensor_size_t>(configuration(), "rows", 8), 1, 32);
                const auto kcols = clamp(from_params<tensor_size_t>(configuration(), "cols", 8), 1, 32);
                const auto kconn = tensor_size_t(1);//clamp(from_params<tensor_size_t>(configuration(), "conn", 1), 1, 16);

                const auto orows = irows - krows + 1;
                const auto ocols = icols - kcols + 1;

                // check convolution size
                if (irows < krows || icols < kcols)
                {
                        log_error() << "convolution layer: invalid size (" << idims << "x" << irows << "x" << icols
                                    << ") -> (" << odims << "x" << krows << "x" << kcols << ")!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // check input connectivity factor
                if ((idims % kconn) || (odims % kconn))
                {
                        log_error() << "convolution layer: invalid input connectivity factor!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // resize buffers
                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_kdata.resize(idims, odims, krows, kcols);
                m_bdata.resize(odims);
                m_kconn = kconn;

                m_toeiodata.resize(krows * kcols, orows * ocols);
                m_toeikdata.resize(orows * ocols, krows * kcols);
                m_toeokdata.resize(krows * kcols, irows * icols);

                m_toeodata.resize(odims, orows * ocols);
                m_toekdata.resize(idims, krows * kcols);

                return psize();
        }

        void conv_layer_toeplitz_t::zero_params()
        {
                tensor::set_zero(m_kdata, m_bdata);
        }

        void conv_layer_toeplitz_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(random_t<scalar_t>(min, max), m_kdata, m_bdata);
        }

        scalar_t* conv_layer_toeplitz_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_kdata, m_bdata);
        }

        const scalar_t* conv_layer_toeplitz_t::load_params(const scalar_t* params)
        {
                return tensor::from_array(params, m_kdata, m_bdata);
        }

        const tensor3d_t& conv_layer_toeplitz_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() == input.size<1>());
                assert(icols() == input.size<2>());

                m_idata = input;

                // convolution
                m_odata.setZero();

                /*tensor::conv2d_dyn_t op;
                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        for (tensor_size_t o = 0; o < odims(); ++ o)
                        {
                                op(m_idata.matrix(i), m_kdata.matrix(i, o), m_odata.matrix(o));
                        }
                }*/

                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        make_conv(m_idata.matrix(i), krows(), kcols(), m_toeiodata);

                        const auto kdata = tensor::map_matrix(m_kdata.planeData(i, 0), odims(), krows() * kcols());
                        tensor::map_matrix(m_odata.data(), odims(), orows() * ocols()) += kdata * m_toeiodata;
                }

                // +bias
                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        m_odata.vector(o).array() += m_bdata(o);
                }

                return m_odata;
        }

        const tensor3d_t& conv_layer_toeplitz_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                m_idata.setZero();

                /*tensor::corr2d_dyn_t op;
                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        for (tensor_size_t o = 0; o < odims(); ++ o)
                        {
                                op(m_odata.matrix(o), m_kdata.matrix(i, o), m_idata.matrix(i));
                        }
                }*/

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        make_corr(m_odata.matrix(o), krows(), kcols(), m_toeokdata);

                        for (tensor_size_t i = 0; i < idims(); ++ i)
                        {
                                m_toekdata.row(i) = m_kdata.vector(i, o);
                        }

                        //const auto kdata = tensor::map_matrix(m_kdata.planeData(o, 0), idims(), krows() * kcols());
                        tensor::map_matrix(m_idata.data(), idims(), irows() * icols()) += m_toekdata * m_toeokdata;
                }

                return m_idata;
        }

        void conv_layer_toeplitz_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                // wrt convolution
                auto gkdata = tensor::map_tensor(gradient, m_kdata.size<0>(), m_kdata.size<1>(), krows(), kcols());
                //gkdata.setZero();

                /*
                tensor::conv2d_dyn_t op;
                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        for (tensor_size_t o = 0; o < odims(); ++ o)
                        {
                                op(m_idata.matrix(i), m_odata.matrix(o), gkdata.matrix(i, o));
                        }
                }*/

                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        make_conv(m_idata.matrix(i), orows(), ocols(), m_toeikdata);

                        const auto odata = tensor::map_matrix(m_odata.data(), odims(), orows() * ocols());
                        tensor::map_matrix(gkdata.planeData(i, 0), odims(), krows() * kcols()) = odata * m_toeikdata;
                }

                // wrt bias
                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        gradient[m_kdata.size() + o] = m_odata.vector(o).sum();
                }
        }
}


