#pragma once

#include "affine_params.h"
#include "tensor/numeric.h"

namespace nano
{
        ///
        /// \brief affine transformation with 4D input and output tensors using
        ///     level-3 Blas calls (thus processing all samples at once).
        ///
        /// parameters:
        ///     idata: 4D input tensor (count x imaps x irows x icols, with isize = imaps x irows x icols)
        ///     wdata: weight matrix (osize x isize)
        ///     bdata: bias vector (osize)
        ///     odata: 4D output tensor (count x omaps x orows x ocols, with osize = omaps x orows x ocols)
        ///
        /// operation:
        ///     odata = wdata * idata + bdata
        ///
        class affine4d_t
        {
        public:
                ///
                /// \brief constructor
                ///
                explicit affine4d_t(const affine_params_t& params = affine_params_t()) :
                        m_params(params) {}

                ///
                /// \brief output
                ///
                template <typename tidata, typename twdata, typename tbdata, typename todata>
                void output(const tidata& idata, const twdata& wdata, const tbdata& bdata, todata&& odata) const;

                ///
                /// \brief gradient wrt inputs
                ///
                template <typename tidata, typename twdata, typename tbdata, typename todata>
                void ginput(tidata&& idata, const twdata& wdata, const tbdata& bdata, const todata& odata) const;

                ///
                /// \brief accumulate the gradient wrt parameters (weights and bias)
                ///
                template <typename tidata, typename twdata, typename tbdata, typename todata>
                void gparam(const tidata& idata, twdata&& wdata, tbdata&& bdata, const todata& odata) const;

                ///
                /// \brief parameters
                ///
                const affine_params_t& params() const { return m_params; }

        private:

                // attributes
                affine_params_t         m_params;
        };

        template <typename tidata, typename twdata, typename tbdata, typename todata>
        void affine4d_t::output(const tidata& idata, const twdata& wdata, const tbdata& bdata, todata&& odata) const
        {
                assert(m_params.valid(idata, wdata, bdata, odata));

                const auto count = idata.template size<0>();
                const auto isize = m_params.isize();
                const auto osize = m_params.osize();

                auto midata = idata.reshape(count, isize).matrix();
                auto modata = odata.reshape(count, osize).matrix();

                modata.noalias() = (midata * wdata.transpose()).rowwise() + bdata.transpose();
        }

        template <typename tidata, typename twdata, typename tbdata, typename todata>
        void affine4d_t::ginput(tidata&& idata, const twdata& wdata, const tbdata& bdata, const todata& odata) const
        {
                assert(m_params.valid(idata, wdata, bdata, odata));
                NANO_UNUSED1_RELEASE(bdata);

                const auto count = idata.template size<0>();
                const auto isize = m_params.isize();
                const auto osize = m_params.osize();

                auto midata = idata.reshape(count, isize).matrix();
                auto modata = odata.reshape(count, osize).matrix();

                midata.transpose().noalias() = wdata.transpose() * modata.transpose();
        }

        template <typename tidata, typename twdata, typename tbdata, typename todata>
        void affine4d_t::gparam(const tidata& idata, twdata&& wdata, tbdata&& bdata, const todata& odata) const
        {
                assert(m_params.valid(idata, wdata, bdata, odata));

                const auto count = idata.template size<0>();
                const auto isize = m_params.isize();
                const auto osize = m_params.osize();

                auto midata = idata.reshape(count, isize).matrix();
                auto modata = odata.reshape(count, osize).matrix();

                wdata.noalias() = modata.transpose() * midata;
                bdata.noalias() = modata.colwise().sum();
        }
}
