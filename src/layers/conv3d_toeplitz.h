#pragma once

#include <cassert>
#include "conv3d_params.h"

namespace nano
{
        ///
        /// \brief Toeplitz-based implementation of the 3D convolution:
        ///     convolutions & correlations are written as matrix multiplications.
        /// NB: requires extra buffers.
        ///
        class conv3d_toeplitz_t
        {
        public:
                ///
                /// \brief constructor
                ///
                conv3d_toeplitz_t(const conv3d_params_t& params = conv3d_params_t());

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

                template <typename timatrix, typename tomatrix>
                void make_toeplitz_output(const timatrix& imat, tomatrix&& omat) const;

                template <typename timatrix, typename tkmatrix>
                void make_toeplitz_gparam(const timatrix& imat, tkmatrix&& kmat) const;

                template <typename tomatrix, typename timatrix>
                void make_toeplitz_ginput(const tomatrix& omat, timatrix&& imat) const;

                // attributes
                conv3d_params_t         m_params;
                mutable tensor3d_t      m_idata_toe;
                // todo: maybe it is possible to not store the kernel!
                mutable tensor4d_t      m_kdata_inv;
                // todo: these should be removed! use directly Eigen calls to map the output buffers!
                mutable matrix_t        m_toe_oodata;
                mutable matrix_t        m_toe_iodata;
                mutable matrix_t        m_toe_iidata;
                mutable matrix_t        m_toe_kodata;
                mutable matrix_t        m_toe_kkdata;
        };

        inline conv3d_toeplitz_t::conv3d_toeplitz_t(const conv3d_params_t& params) :
                m_params(params)
        {
                const auto imaps = m_params.imaps(), irows = m_params.irows(), icols = m_params.icols();
                const auto kconn = m_params.kconn(), krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();

                // allocate buffers
                m_idata_toe.resize(imaps, krows * kcols, orows * ocols);
                m_kdata_inv.resize(imaps, omaps / kconn, krows, kcols);

                m_toe_oodata.resize(omaps / kconn, orows * ocols);

                m_toe_iodata.resize(krows * kcols, irows * icols);
                m_toe_iidata.resize(imaps / kconn, irows * icols);

                m_toe_kodata.resize(omaps / kconn, orows * ocols);
                m_toe_kkdata.resize(omaps / kconn, krows * kcols);
        }

        template <typename timatrix, typename tomatrix>
        void conv3d_toeplitz_t::make_toeplitz_output(const timatrix& imat, tomatrix&& omat) const
        {
                const auto orows = m_params.orows(), ocols = m_params.ocols();
                const auto krows = m_params.krows(), kcols = m_params.kcols();
                const auto drows = m_params.kdrow(), dcols = m_params.kdcol();

                assert(omat.rows() == krows * kcols);
                assert(omat.cols() == orows * ocols);

                for (tensor_size_t kr = 0; kr < krows; ++ kr)
                {
                        for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                        {
                                for (tensor_size_t r = 0; r < orows; ++ r)
                                {
                                        for (tensor_size_t c = 0; c < ocols; ++ c)
                                        {
                                                omat(kr * kcols + kc, r * ocols + c) =
                                                imat(r * drows + kr, c * dcols + kc);
                                        }
                                }
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_toeplitz_t::output(const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata&& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));
                NANO_UNUSED1_RELEASE(kdata);

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn();
                const auto omaps = m_params.omaps();

                for (tensor_size_t o = 0; o < omaps; ++ o)
                {
                        odata.vector(o).setConstant(bdata(o));
                }

                for (tensor_size_t i = 0; i < imaps; ++ i)
                {
                        for (tensor_size_t o = i % kconn, ok = 0; o < omaps; o += kconn, ++ ok)
                        {
                                m_kdata_inv.vector(i, ok) = kdata.vector(o, i / kconn);
                        }
                }

                for (tensor_size_t i = 0; i < imaps; ++ i)
                {
                        make_toeplitz_output(idata.matrix(i), m_idata_toe.matrix(i));

                        m_toe_oodata.noalias() =
                                map_matrix(m_kdata_inv.planeData(i, 0), omaps / kconn, m_kdata_inv.planeSize()) *
                                m_idata_toe.matrix(i);

                        for (tensor_size_t o = i % kconn, ok = 0; o < omaps; o += kconn, ++ ok)
                        {
                                odata.vector(o) += m_toe_oodata.row(ok);
                        }
                }
        }

        template <typename tomatrix, typename timatrix>
        void conv3d_toeplitz_t::make_toeplitz_ginput(const tomatrix& omat, timatrix&& imat) const
        {
                const auto orows = m_params.orows(), ocols = m_params.ocols();
                const auto krows = m_params.krows(), kcols = m_params.kcols();
                const auto drows = m_params.kdrow(), dcols = m_params.kdcol();
                const auto irows = m_params.irows(), icols = m_params.icols();

                assert(imat.rows() == krows * kcols);
                assert(imat.cols() == irows * icols);
                NANO_UNUSED1_RELEASE(irows);

                imat.setZero();
                for (tensor_size_t kr = 0; kr < krows; ++ kr)
                {
                        for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                        {
                                for (tensor_size_t r = 0; r < orows; ++ r)
                                {
                                        for (tensor_size_t c = 0; c < ocols; ++ c)
                                        {
                                                imat(kr * kcols + kc, (r * drows + kr) * icols + c * dcols + kc) +=
                                                omat(r, c);
                                        }
                                }
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_toeplitz_t::ginput(tidata&& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));
                NANO_UNUSED1_RELEASE(bdata);

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn();
                const auto omaps = m_params.omaps();

                idata.setZero();
                for (tensor_size_t o = 0; o < omaps; ++ o)
                {
                        make_toeplitz_ginput(odata.matrix(o), m_toe_iodata);

                        m_toe_iidata.noalias() =
                                map_matrix(kdata.planeData(o, 0), imaps / kconn, kdata.planeSize()) *
                                m_toe_iodata;

                        for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                        {
                                idata.vector(i) += m_toe_iidata.row(ik);
                        }
                }
        }

        template <typename timatrix, typename tkmatrix>
        void conv3d_toeplitz_t::make_toeplitz_gparam(const timatrix& imat, tkmatrix&& kmat) const
        {
                const auto orows = m_params.orows(), ocols = m_params.ocols();
                const auto krows = m_params.krows(), kcols = m_params.kcols();
                const auto drows = m_params.kdrow(), dcols = m_params.kdcol();

                assert(kmat.rows() == orows * ocols);
                assert(kmat.cols() == krows * kcols);

                for (tensor_size_t r = 0; r < orows; ++ r)
                {
                        for (tensor_size_t c = 0; c < ocols; ++ c)
                        {
                                for (tensor_size_t kr = 0; kr < krows; ++ kr)
                                {
                                        for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                        {
                                                kmat(r * ocols + c, kr * kcols + kc) =
                                                imat(r * drows + kr, c * dcols + kc);
                                        }
                                }
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_toeplitz_t::gparam(const tidata& idata, tkdata&& kdata, tbdata&& bdata, const todata& odata) const
        {
                assert(m_params.valid_idata(idata));
                assert(m_params.valid_kdata(kdata));
                assert(m_params.valid_bdata(bdata));
                assert(m_params.valid_odata(odata));
                NANO_UNUSED1_RELEASE(idata);

                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn();
                const auto omaps = m_params.omaps();

                for (tensor_size_t o = 0; o < omaps; ++ o)
                {
                        bdata(o) = odata.vector(o).sum();
                }

                for (tensor_size_t i = 0; i < imaps; ++ i)
                {
                        for (tensor_size_t o = i % kconn, ok = 0; o < omaps; o += kconn, ++ ok)
                        {
                                m_toe_kodata.row(ok) = odata.vector(o);
                        }

                        m_toe_kkdata.noalias() = m_toe_kodata * m_idata_toe.matrix(i).transpose();

                        for (tensor_size_t o = i % kconn, ok = 0; o < omaps; o += kconn, ++ ok)
                        {
                                kdata.vector(o, i / kconn) = m_toe_kkdata.row(ok);
                        }
                }
        }
}
