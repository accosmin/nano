#include "math/clamp.hpp"
#include "math/random.hpp"
#include "logger.h"
#include "tensor/numeric.hpp"
#include "text/to_string.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"
#include "convolution_toeplitz.h"

namespace nano
{
        template <typename timatrix, typename tsize, typename tomatrix>
        static void make_conv(const timatrix& imat, const tsize krows, const tsize kcols, tomatrix&& omat)
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
                                auto orow = omat.row(kr * kcols + kc);

                                for (tsize r = 0; r < orows; ++ r)
                                {
                                        orow.segment(r * ocols, ocols) =
                                        imat.row(r + kr).segment(kc, ocols);
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
                                auto irow = imat.row(kr * kcols + kc);
                                const auto offset = kr * icols + kc;

                                for (tsize r = 0; r < orows; ++ r)
                                {
                                        irow.segment(offset + r * icols, ocols) += omat.row(r);
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

                const auto odims = clamp(from_params<tensor_size_t>(config(), "dims"), 1, 256);
                const auto krows = clamp(from_params<tensor_size_t>(config(), "rows"), 1, 32);
                const auto kcols = clamp(from_params<tensor_size_t>(config(), "cols"), 1, 32);
                const auto kconn = clamp(from_params<tensor_size_t>(config(), "conn"), 1, 16);

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
                m_kdata.resize(odims, idims / kconn, krows, kcols);
                m_bdata.resize(odims);
                m_kconn = kconn;

                const auto alloc_matrices = [] (matrices_t& mats, const auto count, const auto rows, const auto cols)
                {
                        mats.resize(static_cast<size_t>(count));
                        for (auto& mat : mats)
                        {
                                mat.resize(rows, cols);
                        }
                };

                m_toe_oidata.resize(krows * kcols, orows * ocols);
                alloc_matrices(m_toe_okdata, idims, odims / kconn, krows * kcols);
                m_toe_oodata.resize(odims / kconn, orows * ocols);

                m_toe_iodata.resize(krows * kcols, irows * icols);
                alloc_matrices(m_toe_ikdata, odims, idims / kconn, krows * kcols);
                m_toe_iidata.resize(idims / kconn, irows * icols);

                m_toe_kidata.resize(orows * ocols, krows * kcols);
                m_toe_kodata.resize(odims / kconn, orows * ocols);
                m_toe_kkdata.resize(odims / kconn, krows * kcols);

                return psize();
        }

        void conv_layer_toeplitz_t::zero_params()
        {
                tensor::set_zero(m_kdata, m_bdata);
                params_changed();
        }

        void conv_layer_toeplitz_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(random_t<scalar_t>(min, max), m_kdata, m_bdata);
                params_changed();
        }

        scalar_t* conv_layer_toeplitz_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_kdata, m_bdata);
        }

        const scalar_t* conv_layer_toeplitz_t::load_params(const scalar_t* params)
        {
                auto ret = tensor::from_array(params, m_kdata, m_bdata);
                params_changed();
                return ret;
        }

        void conv_layer_toeplitz_t::params_changed()
        {
                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        auto& okdata = m_toe_okdata[static_cast<size_t>(i)];
                        for (tensor_size_t o = (i % kconn()), ok = 0; o < odims(); ++ ok, o += kconn())
                        {
                                okdata.row(ok) = m_kdata.vector(o, i / kconn());
                        }
                }

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        auto& ikdata = m_toe_ikdata[static_cast<size_t>(o)];
                        for (tensor_size_t i = (o % kconn()), ik = 0; i < idims(); ++ ik, i += kconn())
                        {
                                ikdata.row(ik) = m_kdata.vector(o, ik);
                        }
                }
        }

        const tensor3d_t& conv_layer_toeplitz_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() == input.size<1>());
                assert(icols() == input.size<2>());

                m_idata = input;

                // convolution
                m_odata.setZero();

                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        make_conv(m_idata.matrix(i), krows(), kcols(), m_toe_oidata);
                        m_toe_oodata.noalias() = m_toe_okdata[static_cast<size_t>(i)] * m_toe_oidata;

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

        const tensor3d_t& conv_layer_toeplitz_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                // correlation
                m_idata.setZero();

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        make_corr(m_odata.matrix(o), krows(), kcols(), m_toe_iodata);
                        m_toe_iidata.noalias() = m_toe_ikdata[static_cast<size_t>(o)] * m_toe_iodata;

                        for (tensor_size_t i = (o % kconn()), ik = 0; i < idims(); ++ ik, i += kconn())
                        {
                                m_idata.vector(i) += m_toe_iidata.row(ik);
                        }
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
                gkdata.setZero();

                for (tensor_size_t i = 0; i < idims(); ++ i)
                {
                        make_conv(m_idata.matrix(i), orows(), ocols(), m_toe_kidata);
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


