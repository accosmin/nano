#include "logger.h"
#include "convolution.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "tensor/numeric.hpp"
#include "text/to_string.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"

namespace nano
{
        template <typename timatrix, typename tsize, typename tomatrix>
        static void make_conv(const timatrix& imat,
                const tsize krows, const tsize kcols,
                const tsize drows, const tsize dcols,
                tomatrix&& omat)
        {
                const tsize irows = imat.rows();
                const tsize icols = imat.cols();
                const tsize orows = (irows - krows + 1) / drows;
                const tsize ocols = (icols - kcols + 1) / dcols;

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
                                                omat(kr * kcols + kc, r * ocols + c) =
                                                imat(r + kr, kc + c);
                                        }
                                }
                        }
                }
        }

        template <typename tomatrix, typename tsize, typename timatrix>
        static void make_corr(const tomatrix& omat,
                const tsize krows, const tsize kcols,
                const tsize drows, const tsize dcols,
                timatrix& imat)
        {
                const tsize orows = omat.rows();
                const tsize ocols = omat.cols();
                const tsize irows = orows * drows + krows - 1;
                const tsize icols = ocols * dcols + kcols - 1;

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

        convolution_layer_t::convolution_layer_t(const string_t& parameters) :
                layer_t(parameters),
                m_kconn(1), m_drows(1), m_dcols(1)
        {
        }

        tensor_size_t convolution_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = clamp(from_params<tensor_size_t>(config(), "dims"), 1, 256);
                const auto krows = clamp(from_params<tensor_size_t>(config(), "rows"), 1, 32);
                const auto kcols = clamp(from_params<tensor_size_t>(config(), "cols"), 1, 32);
                const auto kconn = clamp(from_params<tensor_size_t>(config(), "conn"), 1, 16);
                const auto drows = clamp(from_params<tensor_size_t>(config(), "drow"), 1, 8);
                const auto dcols = clamp(from_params<tensor_size_t>(config(), "dcol"), 1, 8);

                const auto orows = (irows - krows + 1) / drows;
                const auto ocols = (icols - kcols + 1) / dcols;

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

                // check stride
                if (2 * drows >= krows || 2 * dcols >= kcols)
                {
                        log_error() << "convolution layer: invalid stride - it should be less than half the kernel size!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // resize buffers
                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_kdata.resize(odims, idims / kconn, krows, kcols);
                m_bdata.resize(odims);
                m_kconn = kconn;
                m_drows = drows;
                m_dcols = dcols;

                m_toe_oidata.resize(krows * kcols, orows * ocols);
                m_toe_okdata.resize(odims / kconn, krows * kcols);
                m_toe_oodata.resize(odims / kconn, orows * ocols);

                m_toe_iodata.resize(krows * kcols, irows * icols);
                m_toe_ikdata.resize(idims / kconn, krows * kcols);
                m_toe_iidata.resize(idims / kconn, irows * icols);

                m_toe_kidata.resize(orows * ocols, krows * kcols);
                m_toe_kodata.resize(odims / kconn, orows * ocols);
                m_toe_kkdata.resize(odims / kconn, krows * kcols);

                return psize();
        }

        void convolution_layer_t::zero_params()
        {
                tensor::set_zero(m_kdata, m_bdata);
        }

        void convolution_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(random_t<scalar_t>(min, max), m_kdata, m_bdata);
        }

        scalar_t* convolution_layer_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_kdata, m_bdata);
        }

        const scalar_t* convolution_layer_t::load_params(const scalar_t* params)
        {
                return tensor::from_array(params, m_kdata, m_bdata);
        }

        const tensor3d_t& convolution_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() == input.size<1>());
                assert(icols() == input.size<2>());

                m_idata = input;

                // convolution
                m_odata.setZero();

                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        make_conv(m_idata.matrix(i), krows(), kcols(), m_drows, m_dcols, m_toe_oidata);
                        for (tensor_size_t o = (i % kconn()), ok = 0; o < odims(); ++ ok, o += kconn())
                        {
                                m_toe_okdata.row(ok) = m_kdata.vector(o, i / kconn());
                        }

                        m_toe_oodata.noalias() = m_toe_okdata * m_toe_oidata;

                        for (tensor_size_t o = (i % kconn()), ok = 0; o < odims(); ++ ok, o += kconn())
                        {
                                m_odata.vector(o) += m_toe_oodata.row(ok);
                        }
                }

                // +bias
                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        m_odata.vector(o).array() += m_bdata(o);
                }

                return m_odata;
        }

        const tensor3d_t& convolution_layer_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                // correlation
                m_idata.setZero();

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        make_corr(m_odata.matrix(o), krows(), kcols(), m_drows, m_dcols, m_toe_iodata);
                        for (tensor_size_t i = (o % kconn()), ik = 0; i < idims(); ++ ik, i += kconn())
                        {
                                m_toe_ikdata.row(ik) = m_kdata.vector(o, ik);
                        }

                        m_toe_iidata.noalias() = m_toe_ikdata * m_toe_iodata;

                        for (tensor_size_t i = (o % kconn()), ik = 0; i < idims(); ++ ik, i += kconn())
                        {
                                m_idata.vector(i) += m_toe_iidata.row(ik);
                        }
                }

                return m_idata;
        }

        void convolution_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                // wrt convolution
                auto gkdata = tensor::map_tensor(gradient, m_kdata.size<0>(), m_kdata.size<1>(), krows(), kcols());
                gkdata.setZero();

                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        make_conv(m_idata.matrix(i), orows(), ocols(), m_drows, m_dcols, m_toe_kidata);
                        for (tensor_size_t o = (i % kconn()), ok = 0; o < odims(); ++ ok, o += kconn())
                        {
                                m_toe_kodata.row(ok) = m_odata.vector(o);
                        }

                        m_toe_kkdata.noalias() = m_toe_kodata * m_toe_kidata;
                        for (tensor_size_t o = (i % kconn()), ok = 0; o < odims(); ++ ok, o += kconn())
                        {
                                gkdata.vector(o, i / kconn()) += m_toe_kkdata.row(ok);
                        }
                }

                // wrt bias
                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        gradient[m_kdata.size() + o] = m_odata.vector(o).sum();
                }
        }
}


