#pragma once

#include "conv_utils.h"
#include "conv3d_params.h"

namespace nano
{
        ///
        /// \brief convolution transformation with 4D input and output tensors using
        ///     efficient level-3 BLAS calls.
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
        /// operation:
        ///     odata(o) = sum(i, conv2d(idata(i), kdata(o, i))) + bdata(o)
        ///
        class conv4d_t
        {
        public:
                ///
                /// \brief constructor
                ///
                explicit conv4d_t(const conv3d_params_t& params = conv3d_params_t());

                ///
                /// \brief output
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                void output(const tidata&, const tkdata&, const tbdata&, todata&&);

                ///
                /// \brief gradient wrt inputs
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                void ginput(tidata&&, const tkdata&, const tbdata&, const todata&);

                ///
                /// \brief gradient wrt parameters (convolution kernels and bias)
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                void gparam(const tidata&, tkdata&&, tbdata&&, const todata& odata);

                ///
                /// \brief parameters
                ///
                const conv3d_params_t& params() const { return m_params; }

        private:

                // attributes
                conv3d_params_t m_params;
                matrix_t        m_okdata;       ///< buffer: (omaps, imaps x krows x kcols)
                matrix_t        m_xkdata;       ///< buffer: (omaps, imaps x krows x kcols)
                tensor3d_t      m_kodata;       ///< buffer: (count, imaps x krows x kcols, orows x ocols)
                matrix_t        m_kxdata;       ///< buffer: (imaps x krows x kcols, orows x ocols)
        };

        inline conv4d_t::conv4d_t(const conv3d_params_t& params) :
                m_params(params)
        {
                const auto imaps = m_params.imaps();
                const auto krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();

                // allocate buffers
                m_okdata.resize(omaps, imaps * krows * kcols);
                m_xkdata.resize(omaps, imaps * krows * kcols);
                m_kxdata.resize(imaps * krows * kcols, orows * ocols);
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv4d_t::output(const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata&& odata)
        {
                assert(m_params.valid(idata, kdata, bdata, odata));

                const auto count = idata.template size<0>();
                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn(), krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();
                const auto drows = m_params.kdrow(), dcols = m_params.kdcol();

                switch (kconn)
                {
                case 1:
                        m_okdata = kdata.reshape(omaps, imaps * krows * kcols).matrix();
                        break;

                default:
                        m_okdata.setZero();
                        for (tensor_size_t o = 0; o < omaps; ++ o)
                        {
                                for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                                {
                                        m_okdata.row(o).segment(i * krows * kcols, krows * kcols) =
                                        kdata.vector(o, ik);
                                }
                        }
                        break;
                }

//                output[x] = kernel * input[x]
//
//                oodata                  = okdata *                         kodata
//                (omaps, orows * ocols)  = (omaps, imaps * krows * kcols) x (imaps * krows * kcols, orows * ocols)

                m_kodata.resize(count, imaps * krows * kcols, orows * ocols);
                for (tensor_size_t x = 0; x < count; ++ x)
                {
                        auto xidata = idata.tensor(x);
                        auto xodata = odata.tensor(x).reshape(omaps, orows * ocols).matrix();
                        auto kodata = m_kodata.matrix(x);

                        // bias
                        xodata.colwise() = bdata;

                        // +convolution
                        for (tensor_size_t i = 0; i < imaps; ++ i)
                        {
                                img2col(xidata.matrix(i), orows, ocols, krows, kcols, drows, dcols,
                                         map_matrix(kodata.row(i * krows * kcols).data(),
                                                    krows * kcols, orows * ocols));
                        }

                        xodata.noalias() += m_okdata * kodata;
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv4d_t::ginput(tidata&& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata)
        {
                assert(m_params.valid(idata, kdata, bdata, odata));
                NANO_UNUSED2(kdata, bdata);

                const auto count = idata.template size<0>();
                const auto imaps = m_params.imaps();
                const auto krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();
                const auto drows = m_params.kdrow(), dcols = m_params.kdcol();

                for (tensor_size_t x = 0; x < count; ++ x)
                {
                        auto xidata = idata.tensor(x);
                        auto xodata = odata.tensor(x);

                        m_kxdata.noalias() = m_okdata.transpose() * xodata.reshape(omaps, orows * ocols).matrix();

                        xidata.zero();
                        for (tensor_size_t i = 0; i < imaps; ++ i)
                        {
                                col2img(xidata.matrix(i), orows, ocols, krows, kcols, drows, dcols,
                                        map_matrix(m_kxdata.row(i * krows * kcols).data(),
                                                   krows * kcols, orows * ocols));
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv4d_t::gparam(const tidata& idata, tkdata&& kdata, tbdata&& bdata, const todata& odata)
        {
                assert(m_params.valid(idata, kdata, bdata, odata));

                const auto count = idata.template size<0>();
                const auto imaps = m_params.imaps();
                const auto kconn = m_params.kconn(), krows = m_params.krows(), kcols = m_params.kcols();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();

                kdata.setZero();
                bdata.setZero();

                assert(m_kodata.size<0>() == count);
                assert(m_kodata.size<1>() == imaps * krows * kcols);
                assert(m_kodata.size<2>() == orows * ocols);
                for (tensor_size_t x = 0; x < count; ++ x)
                {
                        auto xodata = odata.tensor(x).reshape(omaps, orows * ocols).matrix();
                        auto kodata = m_kodata.matrix(x);

                        // bias
                        bdata += xodata.rowwise().sum();

                        // convolution
                        m_xkdata.noalias() = xodata * kodata.transpose();

                        switch (kconn)
                        {
                        case 1:
                                kdata.reshape(omaps, imaps * krows * kcols).matrix().noalias() += m_xkdata;
                                break;

                        default:
                                for (tensor_size_t o = 0; o < omaps; ++ o)
                                {
                                        for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                                        {
                                                kdata.vector(o, ik) +=
                                                m_xkdata.row(o).segment(i * krows * kcols, krows * kcols);
                                        }
                                }
                                break;
                        }
                }
        }
}
