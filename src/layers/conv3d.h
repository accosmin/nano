#pragma once

#include "conv_utils.h"
#include "conv_params.h"

namespace nano
{
        ///
        /// \brief 3D convolution transformation with 4D input and output tensors using
        ///     unrolled & not vectorized looping through pixels.
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
        struct conv3d_t
        {
                ///
                /// \brief constructor
                ///
                explicit conv3d_t(const conv_params_t& params = conv_params_t()) :
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
                /// \brief accumulate the gradient wrt parameters (convolution kernels and bias)
                ///
                template <typename tidata, typename tkdata, typename tbdata, typename todata>
                void gparam(const tidata&, tkdata&&, tbdata&&, const todata& odata) const;

                ///
                /// \brief parameters
                ///
                const conv_params_t& params() const { return m_params; }

        private:

                // attributes
                conv_params_t   m_params;
        };

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_t::output(const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata&& odata) const
        {
                assert(m_params.valid(idata, kdata, bdata, odata));

                const auto count = idata.template size<0>();
                const auto imaps = m_params.imaps();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();
                const auto kconn = m_params.kconn(), kdrow = m_params.kdrow(), kdcol = m_params.kdcol();

                for (tensor_size_t x = 0; x < count; ++ x)
                {
                        auto xidata = idata.tensor(x);
                        auto xodata = odata.tensor(x);

                        // bias
                        xodata.reshape(omaps, orows * ocols).matrix().colwise() = bdata;

                        // + convolution
                        for (tensor_size_t o = 0; o < omaps; ++ o)
                        {
                                for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                                {
                                        convo2d(xidata.matrix(i), kdata.matrix(o, ik), kdrow, kdcol, xodata.matrix(o));
                                }
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_t::ginput(tidata&& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata) const
        {
                assert(m_params.valid(idata, kdata, bdata, odata));

                const auto count = idata.template size<0>();
                const auto imaps = m_params.imaps();
                const auto omaps = m_params.omaps();
                const auto kconn = m_params.kconn(), kdrow = m_params.kdrow(), kdcol = m_params.kdcol();

                for (tensor_size_t x = 0; x < count; ++ x)
                {
                        auto xidata = idata.tensor(x);
                        auto xodata = odata.tensor(x);

                        xidata.setZero();
                        for (tensor_size_t i = 0; i < imaps; ++ i)
                        {
                                for (tensor_size_t o = i % kconn, ik = i / kconn; o < omaps; o += kconn)
                                {
                                        convi2d(xidata.matrix(i), kdata.matrix(o, ik), kdrow, kdcol, xodata.matrix(o));
                                }
                        }
                }
        }

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        void conv3d_t::gparam(const tidata& idata, tkdata&& kdata, tbdata&& bdata, const todata& odata) const
        {
                assert(m_params.valid(idata, kdata, bdata, odata));

                const auto count = idata.template size<0>();
                const auto imaps = m_params.imaps();
                const auto omaps = m_params.omaps(), orows = m_params.orows(), ocols = m_params.ocols();
                const auto kconn = m_params.kconn(), kdrow = m_params.kdrow(), kdcol = m_params.kdcol();

                kdata.setZero();
                bdata.setZero();

                for (tensor_size_t x = 0; x < count; ++ x)
                {
                        auto xidata = idata.tensor(x);
                        auto xodata = odata.tensor(x);

                        // bias
                        bdata += xodata.reshape(omaps, orows * ocols).matrix().rowwise().sum();

                        // convolution
                        for (tensor_size_t o = 0; o < omaps; ++ o)
                        {
                                for (tensor_size_t i = o % kconn, ik = 0; i < imaps; i += kconn, ++ ik)
                                {
                                        convk2d(xidata.matrix(i), kdata.matrix(o, ik), kdrow, kdcol, xodata.matrix(o));
                                }
                        }
                }
        }
}
