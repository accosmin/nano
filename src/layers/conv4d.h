#pragma once

#include "conv_utils.h"
#include "conv_params.h"

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
                tensor3d_t      m_kodata;       ///< buffer: (count, imaps x krows x kcols, orows x ocols)
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
                m_kxdata.resize(imaps * krows * kcols, orows * ocols);
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        bool conv4d_t::output(const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata&& odata)
        {
                if (m_params.valid(idata, kdata, bdata, odata))
                {
                        const auto count = idata.template size<0>();
                        const auto imaps = m_params.imaps(), isize = m_params.isize();
                        const auto kconn = m_params.kconn(), kdrow = m_params.kdrow(), kdcol = m_params.kdcol(), krows = m_params.krows(), kcols = m_params.kcols();
                        const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols(), osize = m_params.osize();

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

                        m_kodata.resize(count, imaps * krows * kcols, orows * ocols);
                        for (tensor_size_t x = 0; x < count; ++ x)
                        {
                                auto imap = map_tensor(idata.data() + x * isize, m_params.idims());
                                auto omap = map_tensor(odata.data() + x * osize, m_params.odims());
                                auto kodata = m_kodata.matrix(x);

                                // bias
                                map_matrix(omap.data(), omaps, orows * ocols).colwise() = bdata;

                                // +convolution
                                for (tensor_size_t i = 0; i < imaps; ++ i)
                                {
                                        img2col(imap.matrix(i), orows, ocols, krows, kcols, kdrow, kdcol,
                                                map_matrix(kodata.row(i * krows * kcols).data(),
                                                           krows * kcols, orows * ocols));
                                }

                                map_matrix(omap.data(), omaps, orows * ocols) += m_okdata * kodata;
                        }
                        return true;
                }
                else
                {
                        return false;
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        bool conv4d_t::ginput(tidata&& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata)
        {
                if (m_params.valid(idata, kdata, bdata, odata))
                {
                        const auto count = idata.template size<0>();
                        const auto imaps = m_params.imaps(), isize = m_params.isize();
                        const auto krows = m_params.krows(), kcols = m_params.kcols();
                        const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols(), osize = m_params.osize();
                        const auto drows = m_params.kdrow(), dcols = m_params.kdcol();

                        for (tensor_size_t x = 0; x < count; ++ x)
                        {
                                auto imap = map_tensor(idata.data() + x * isize, m_params.idims());
                                auto omap = map_tensor(odata.data() + x * osize, m_params.odims());

                                m_oodata = map_matrix(omap.data(), omaps, orows * ocols);
                                m_kxdata.noalias() = m_okdata.transpose() * m_oodata;

                                imap.array().setZero();
                                for (tensor_size_t i = 0; i < imaps; ++ i)
                                {
                                        auto imat = imap.matrix(i);
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
                        return true;
                }
                else
                {
                        return false;
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        bool conv4d_t::gparam(const tidata& idata, tkdata&& kdata, tbdata&& bdata, const todata& odata)
        {
                if (m_params.valid(idata, kdata, bdata, odata))
                {
                        const auto count = idata.template size<0>();
                        const auto imaps = m_params.imaps();
                        const auto kconn = m_params.kconn(), krows = m_params.krows(), kcols = m_params.kcols();
                        const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols(), osize = m_params.osize();

                        kdata.vector().setZero();
                        bdata.setZero();

                        assert(m_kodata.dims() == tensor3d_dims_t(count, imaps * krows * kcols, orows * ocols));
                        for (tensor_size_t x = 0; x < count; ++ x)
                        {
                                auto omap = map_tensor(odata.data() + x * osize, m_params.odims());

                                // bias
                                bdata += map_matrix(omap.data(), omaps, orows * ocols).rowwise().sum();

                                // convolution
                                m_oodata = map_matrix(omap.data(), omaps, orows * ocols);
                                m_xkdata.noalias() = m_oodata * m_kodata.matrix(x).transpose();

                                switch (kconn)
                                {
                                case 1:
                                        map_matrix(kdata.data(), omaps, imaps * krows * kcols) += m_xkdata;
                                        break;

                                default:
                                        for (tensor_size_t o = 0; o < omaps; ++ o)
                                        {
                                                for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                                                {
                                                        kdata.vector(o, ik) += m_xkdata.row(o).segment(i * krows * kcols, krows * kcols);
                                                }
                                        }
                                        break;
                                }
                        }
                        return true;
                }
                else
                {
                        return false;
                }
        }
}
