#pragma once

#include <cassert>
#include "affine_params.h"

namespace nano
{
        ///
        /// \brief affine transformation with 4D input and output tensors using
        ///     level-3 Blas calls (thus processing all samples at once).
        ///
        /// parameters:
        ///     idata: 4D input tensor (count x iplanes x irows x icols, with isize = iplanes x irows x icols)
        ///     wdata: weight matrix (osize x isize)
        ///     bdata: bias vector (osize)
        ///     odata: 4D output tensor (count x oplanes x orows x ocols, with osize = oplanes x orows x ocols)
        ///
        struct affine4d_t
        {
                ///
                /// \brief constructor
                ///
                explicit affine4d_t(const affine_params_t& params = affine_params_t()) :
                        m_params(params) {}

                ///
                /// \brief output
                ///
                template <typename tidata, typename twdata, typename tbdata, typename todata>
                bool output(const tidata& idata, const twdata& wdata, const tbdata& bdata, todata&& odata);

                ///
                /// \brief gradient wrt inputs
                ///
                template <typename tidata, typename twdata, typename tbdata, typename todata>
                bool ginput(tidata&& idata, const twdata& wdata, const tbdata& bdata, const todata& odata);

                ///
                /// \brief accumulate the gradient wrt parameters (weights and bias)
                ///
                template <typename tidata, typename twdata, typename tbdata, typename todata>
                bool gparam(const tidata& idata, twdata&& wdata, tbdata&& bdata, const todata& odata);

                ///
                /// \brief parameters
                ///
                const affine_params_t& params() const { return m_params; }

        private:

                // attributes
                affine_params_t         m_params;
                matrix_t                m_idata;        ///< (aligned) input buffer
                matrix_t                m_odata;        ///< (aligned) output buffer
        };

        template <typename tidata, typename twdata, typename tbdata, typename todata>
        bool affine4d_t::output(const tidata& idata, const twdata& wdata, const tbdata& bdata, todata&& odata)
        {
                if (m_params.valid(idata, wdata, bdata, odata))
                {
                        const auto count = idata.template size<0>();
                        const auto isize = m_params.isize();
                        const auto osize = m_params.osize();

                        m_idata.resize(count, isize);
                        m_odata.resize(count, osize);

                        m_idata = map_matrix(idata.data(), count, isize);
                        m_odata = m_idata * wdata.transpose();
                        m_odata.rowwise() += bdata.transpose();
                        map_matrix(odata.data(), count, osize) = m_odata;
                        return true;
                }
                else
                {
                        return false;
                }
        }

        template <typename tidata, typename twdata, typename tbdata, typename todata>
        bool affine4d_t::ginput(tidata&& idata, const twdata& wdata, const tbdata& bdata, const todata& odata)
        {
                NANO_UNUSED1_RELEASE(bdata);

                if (m_params.valid(idata, wdata, bdata, odata))
                {
                        const auto count = idata.template size<0>();
                        const auto isize = m_params.isize();
                        const auto osize = m_params.osize();

                        m_idata.resize(count, isize);
                        m_odata.resize(count, osize);

                        m_odata = map_matrix(odata.data(), count, osize);
                        m_idata = m_odata * wdata;
                        map_matrix(idata.data(), count, isize) = m_idata;
                        return true;
                }
                else
                {
                        return false;
                }
        }

        template <typename tidata, typename twdata, typename tbdata, typename todata>
        bool affine4d_t::gparam(const tidata& idata, twdata&& wdata, tbdata&& bdata, const todata& odata)
        {
                if (m_params.valid(idata, wdata, bdata, odata))
                {
                        const auto count = idata.template size<0>();
                        const auto isize = m_params.isize();
                        const auto osize = m_params.osize();

                        m_idata.resize(count, isize);
                        m_odata.resize(count, osize);

                        m_idata = map_matrix(idata.data(), count, isize);
                        m_odata = map_matrix(odata.data(), count, osize);
                        wdata = m_odata.transpose() * m_idata;
                        bdata = m_odata.colwise().sum();
                        return true;
                }
                else
                {
                        return false;
                }
        }
}
