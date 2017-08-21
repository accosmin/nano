#pragma once

#include "conv3d_utils.h"

namespace nano
{
        ///
        /// \brief convolution transformation with 4D input and output tensors using
        ///     level-3 Blas calls (thus processing all samples at once).
        ///
        /// NB: the 3D convolutions and correlations are replaced with matrix multiplications.
        /// NB: requires extra buffers.
        ///
        /// parameters:
        ///     idata: 4D input tensor (count x imaps x irows x icols, with isize = imaps x irows x icols)
        ///     kdata: convolution kernel (omaps x imaps/kconn x krows x kcols)
        ///     bdata: bias vector (omaps)
        ///     odata: 4D output tensor (count x omaps x orows x ocols, with osize = omaps x orows x ocols)
        ///
        struct conv4d_t
        {
                ///
                /// \brief constructor
                ///
                explicit conv4d_t(const conv_params_t& params = conv_params_t());

                ///
                /// \brief output
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                bool output(const tidata&, const tkdata&, const tbdata&, todata&&);

                ///
                /// \brief gradient wrt inputs
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                bool ginput(tidata&&, const tkdata&, const tbdata&, const todata&);

                ///
                /// \brief gradient wrt parameters (convolution kernels and bias)
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                bool gparam(const tidata&, tkdata&&, tbdata&&, const todata& odata);

                ///
                /// \brief parameters
                ///
                const conv_params_t& params() const { return m_params; }

        private:

                // attributes
                conv_params_t   m_params;
                matrix_t        m_oodata;       ///< buffer: (omaps, orows x ocols)
                matrix_t        m_okdata;       ///< buffer: (omaps, imaps x krows x kcols)
                matrix_t        m_xkdata;       ///< buffer: (omaps, imaps x krows x kcols)
                matrix_t        m_kodata;       ///< buffer: (imaps x krows x kcols, orows x ocols)
                matrix_t        m_kxdata;       ///< buffer: (imaps x krows x kcols, orows x ocols)
        };

        inline conv4d_t::conv4d_t(const conv_params_t& params) :
                m_params(params)
        {
                const auto imaps = m_params.imaps();
                const auto krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();

                // allocate buffers
                m_oodata.resize(omaps, orows * ocols);
                m_okdata.resize(omaps, imaps * krows * kcols);
                m_xkdata.resize(omaps, imaps * krows * kcols);
                m_kodata.resize(imaps * krows * kcols, orows * ocols);
                m_kxdata.resize(imaps * krows * kcols, orows * ocols);
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        bool conv4d_t::output(const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata&& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn(), krows = m_params.krows(), kcols = m_params.kcols(), ksize = krows * kcols;
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols(), osize = orows * ocols;

                // bias
                for (tensor_size_t o = 0; o < omaps; ++ o)
                {
                        odata.vector(o).setConstant(bdata(o));
                }

                // +convolution
                switch (kconn)
                {
                case 1:
                        m_okdata = map_matrix(kdata.data(), omaps, imaps * krows * kcols);
                        break;

                default:
                        m_okdata.setZero();
                        for (tensor_size_t o = 0; o < omaps; ++ o)
                        {
                                for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                                {
                                        m_okdata.row(o).segment(i * krows * kcols, krows * kcols) = kdata.vector(o, ik);
                                }
                        }
                        break;
                }

                for (tensor_size_t i = 0; i < imaps; ++ i)
                {
                        img2col(m_params, idata.matrix(i), map_matrix(m_kodata.data() + i * ksize * osize, ksize, osize));
                }

                m_oodata.noalias() = m_okdata * m_kodata;
                map_matrix(odata.data(), omaps, orows * ocols) += m_oodata;
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv4d_t::ginput(tidata&& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));
                NANO_UNUSED1_RELEASE(kdata);
                NANO_UNUSED1_RELEASE(bdata);

                const auto imaps = m_params.imaps();
                const auto krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();
                const auto drows = m_params.kdrow(), dcols = m_params.kdcol();

                m_oodata = map_matrix(odata.data(), omaps, orows * ocols);
                m_kxdata.noalias() = m_okdata.transpose() * m_oodata;

                idata.array().setZero();
                for (tensor_size_t i = 0; i < imaps; ++ i)
                {
                        auto imat = idata.matrix(i);
                        for (tensor_size_t kr = 0; kr < krows; ++ kr)
                        {
                                for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                {
                                        const auto orow = m_kxdata.row(i * krows * kcols + kr * kcols + kc);
                                        for (tensor_size_t r = 0; r < orows; ++ r)
                                        {
                                                for (tensor_size_t c = 0; c < ocols; ++ c)
                                                {
                                                        imat(r * drows + kr, c * dcols + kc) += orow(r * ocols + c);
                                                }
                                        }
                                }
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv4d_t::gparam(const tidata& idata, tkdata&& kdata, tbdata&& bdata, const todata& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));
                NANO_UNUSED1_RELEASE(idata);

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn(), krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();

                // bias
                for (tensor_size_t o = 0; o < omaps; ++ o)
                {
                        bdata(o) = odata.vector(o).sum();
                }

                // convolution
                m_oodata = map_matrix(odata.data(), omaps, orows * ocols);
                m_xkdata.noalias() = m_oodata * m_kodata.transpose();

                switch (kconn)
                {
                case 1:
                        map_matrix(kdata.data(), omaps, imaps * krows * kcols) = m_xkdata;
                        break;

                default:
                        for (tensor_size_t o = 0; o < omaps; ++ o)
                        {
                                for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                                {
                                        kdata.vector(o, ik) = m_xkdata.row(o).segment(i * krows * kcols, krows * kcols);
                                }
                        }
                        break;
                }
        }
}
