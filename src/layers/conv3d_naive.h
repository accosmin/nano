#pragma once

#include <cassert>
#include "conv3d_params.h"

namespace nano
{
        ///
        /// \brief naive implementation of the 3D convolution:
        ///     unrolled & not vectorized looping through pixels.
        ///
        struct conv3d_naive_t
        {
                ///
                /// \brief constructor
                ///
                explicit conv3d_naive_t(const conv3d_params_t& params = conv3d_params_t()) :
                        m_params(params) {}

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
        };

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_naive_t::output(const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata&& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));

                const auto conv2d = [] (const auto& imat, const auto& kmat, const auto dr, const auto dc, auto&& omat)
                {
                        for (tensor_size_t orows = omat.rows(), r = 0; r < orows; ++ r)
                        {
                                for (tensor_size_t ocols = omat.cols(), c = 0; c < ocols; ++ c)
                                {
                                        for (tensor_size_t krows = kmat.rows(), kr = 0; kr < krows; ++ kr)
                                        {
                                                for (tensor_size_t kcols = kmat.cols(), kc = 0; kc < kcols; ++ kc)
                                                {
                                                        omat(r, c) += imat(r * dr + kr, c * dc + kc) * kmat(kr, kc);
                                                }
                                        }
                                }
                        }
                };

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn();
                const auto omaps = m_params.omaps();
                const auto kdrow = m_params.kdrow();
                const auto kdcol = m_params.kdcol();

                for (tensor_size_t o = 0; o < omaps; ++ o)
                {
                        odata.matrix(o).setConstant(bdata(o));

                        for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                        {
                                conv2d(idata.matrix(i), kdata.matrix(o, ik), kdrow, kdcol, odata.matrix(o));
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_naive_t::ginput(tidata&& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));
                NANO_UNUSED1_RELEASE(bdata);

                const auto corr2d = [] (auto&& imat, const auto& kmat, const auto dr, const auto dc, const auto& omat)
                {
                        for (tensor_size_t orows = omat.rows(), r = 0; r < orows; ++ r)
                        {
                                for (tensor_size_t ocols = omat.cols(), c = 0; c < ocols; ++ c)
                                {
                                        for (tensor_size_t krows = kmat.rows(), kr = 0; kr < krows; ++ kr)
                                        {
                                                for (tensor_size_t kcols = kmat.cols(), kc = 0; kc < kcols; ++ kc)
                                                {
                                                        imat(r * dr + kr, c * dc + kc) += omat(r, c) * kmat(kr, kc);
                                                }
                                        }
                                }
                        }
                };

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn();
                const auto omaps = m_params.omaps();
                const auto kdrow = m_params.kdrow();
                const auto kdcol = m_params.kdcol();

                for (tensor_size_t i = 0; i < imaps; ++ i)
                {
                        idata.matrix(i).setZero();

                        for (tensor_size_t o = i % kconn, ok = 0; o < omaps; o += kconn, ++ ok)
                        {
                                corr2d(idata.matrix(i), kdata.matrix(o, i / kconn), kdrow, kdcol, odata.matrix(o));
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_naive_t::gparam(const tidata& idata, tkdata&& kdata, tbdata&& bdata, const todata& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));

                const auto conv2d = [] (const auto& imat, auto&& kmat, const auto dr, const auto dc, const auto& omat)
                {
                        for (tensor_size_t orows = omat.rows(), r = 0; r < orows; ++ r)
                        {
                                for (tensor_size_t ocols = omat.cols(), c = 0; c < ocols; ++ c)
                                {
                                        for (tensor_size_t krows = kmat.rows(), kr = 0; kr < krows; ++ kr)
                                        {
                                                for (tensor_size_t kcols = kmat.cols(), kc = 0; kc < kcols; ++ kc)
                                                {
                                                        kmat(kr, kc) += imat(r * dr + kr, c * dc + kc) * omat(r, c);
                                                }
                                        }
                                }
                        }
                };

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn();
                const auto omaps = m_params.omaps();
                const auto kdrow = m_params.kdrow();
                const auto kdcol = m_params.kdcol();

                for (tensor_size_t o = 0; o < omaps; ++ o)
                {
                        bdata(o) = odata.matrix(o).sum();

                        for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                        {
                                kdata.matrix(o, ik).setZero();
                                conv2d(idata.matrix(i), kdata.matrix(o, ik), kdrow, kdcol, odata.matrix(o));
                        }
                }
        }
}
