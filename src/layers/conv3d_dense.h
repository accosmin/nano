#pragma once

#include <cassert>
#include "conv3d_params.h"

namespace nano
{
        ///
        /// \brief Toeplitz-based implementation of the 3D convolution:
        ///     convolutions & correlations are written as a single matrix multiplication.
        /// NB: requires extra buffers.
        ///
        struct conv3d_dense_t
        {
                ///
                /// \brief constructor
                ///
                explicit conv3d_dense_t(const conv3d_params_t& params = conv3d_params_t());

                ///
                /// \brief output
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                void output(const tidata&, const tkdata&, const tbdata&, todata&&) const;

                ///
                /// \brief gradient wrt inputs
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                void ginput(tidata&&, const tkdata&, const tbdata&, const todata&) const;

                ///
                /// \brief gradient wrt parameters (convolution kernels and bias)
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                void gparam(const tidata&, tkdata&&, tbdata&&, const todata& odata) const;

                ///
                /// \brief parameters
                ///
                const conv3d_params_t& params() const { return m_params; }

        private:

                // attributes
                conv3d_params_t         m_params;
                mutable matrix_t        m_oodata;       ///< buffer: (omaps, orows x ocols)
                mutable matrix_t        m_okdata;       ///< buffer: (omaps, imaps x krows x kcols)
                mutable matrix_t        m_xkdata;       ///< buffer: (omaps, imaps x krows x kcols)
                mutable matrix_t        m_kodata;       ///< buffer: (imaps x krows x kcols, orows x ocols)
                mutable matrix_t        m_kxdata;       ///< buffer: (imaps x krows x kcols, orows x ocols)
        };

        inline conv3d_dense_t::conv3d_dense_t(const conv3d_params_t& params) :
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
        void conv3d_dense_t::output(const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata&& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn(), krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();
                const auto drows = m_params.kdrow(), dcols = m_params.kdcol();

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
                        const auto imat = idata.matrix(i);
                        for (tensor_size_t kr = 0; kr < krows; ++ kr)
                        {
                                for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                {
                                        auto orow = m_kodata.row(i * krows * kcols + kr * kcols + kc);
                                        for (tensor_size_t r = 0; r < orows; ++ r)
                                        {
                                                for (tensor_size_t c = 0; c < ocols; ++ c)
                                                {
                                                        orow(r * ocols + c) = imat(r * drows + kr, c * dcols + kc);
                                                }
                                        }
                                }
                        }
                }

                m_oodata.noalias() = m_okdata * m_kodata;
                map_matrix(odata.data(), omaps, orows * ocols) += m_oodata;
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_dense_t::ginput(tidata&& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata) const
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
        void conv3d_dense_t::gparam(const tidata& idata, tkdata&& kdata, tbdata&& bdata, const todata& odata) const
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
