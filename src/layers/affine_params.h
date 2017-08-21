#pragma once

#include "tensor.h"

namespace nano
{
        ///
        /// \brief parametrizes the affine transformation (y = W * x + b) used by convolution networks or MLPs.
        ///
        struct affine_params_t
        {
                affine_params_t(
                        const tensor_size_t imaps = 0, const tensor_size_t irows = 0, const tensor_size_t icols = 0,
                        const tensor_size_t omaps = 0, const tensor_size_t orows = 0, const tensor_size_t ocols = 0) :
                        m_imaps(imaps), m_irows(irows), m_icols(icols),
                        m_omaps(omaps), m_orows(orows), m_ocols(ocols) {}

                auto imaps() const { return m_imaps; }
                auto irows() const { return m_irows; }
                auto icols() const { return m_icols; }
                auto idims() const { return tensor3d_dims_t{imaps(), irows(), icols()}; }
                auto idims(const tensor_size_t count) const { return tensor4d_dims_t{count, imaps(), irows(), icols()}; }
                auto isize() const { return nano::size(idims()); }

                auto omaps() const { return m_omaps; }
                auto orows() const { return m_orows; }
                auto ocols() const { return m_ocols; }
                auto odims() const { return tensor3d_dims_t{omaps(), orows(), ocols()}; }
                auto odims(const tensor_size_t count) const { return tensor4d_dims_t{count, omaps(), orows(), ocols()}; }
                auto osize() const { return nano::size(odims()); }

                auto psize() const { return isize() * osize() + osize(); }
                auto flops_output() const { return 2 * isize() * osize() + osize(); }
                auto flops_ginput() const { return 2 * isize() * osize(); }
                auto flops_gparam() const { return 2 * isize() * osize() + osize(); }

                auto make_idata(const tensor_size_t count) const { return tensor4d_t(idims(count)); }
                auto make_odata(const tensor_size_t count) const { return tensor4d_t(odims(count)); }
                auto make_wdata() const { return matrix_t(osize(), isize()); }
                auto make_bdata() const { return vector_t(osize()); }

                template <typename tidata, typename twdata, typename tbdata, typename todata>
                bool valid(const tidata&, const twdata&, const tbdata&, const todata&) const;

                bool valid() const { return isize() > 0 && osize() > 0; }

                // attributes
                tensor_size_t   m_imaps, m_irows, m_icols;      ///< input size
                tensor_size_t   m_omaps, m_orows, m_ocols;      ///< output size
        };

        template <typename tidata, typename twdata, typename tbdata, typename todata>
        inline bool affine_params_t::valid(
                const tidata& idata, const twdata& wdata, const tbdata& bdata, const todata& odata) const
        {
                const auto count = idata.template size<0>();
                return  idata.template size<0>() == count &&
                        idata.template size<1>() == imaps() &&
                        idata.template size<2>() == irows() &&
                        idata.template size<3>() == icols() &&
                        wdata.rows() == osize() &&
                        wdata.cols() == isize() &&
                        bdata.size() == osize() &&
                        odata.template size<0>() == count &&
                        odata.template size<1>() == omaps() &&
                        odata.template size<2>() == orows() &&
                        odata.template size<3>() == ocols();
        }

        inline bool operator==(const affine_params_t& params1, const affine_params_t& params2)
        {
                return  params1.m_imaps == params2.m_imaps &&
                        params1.m_irows == params2.m_irows &&
                        params1.m_icols == params2.m_icols &&
                        params1.m_omaps == params2.m_omaps &&
                        params1.m_orows == params2.m_orows &&
                        params1.m_ocols == params2.m_ocols;
        }
}
