#pragma once

#include "tensor.h"

namespace nano
{
        ///
        /// \brief parametrizes the 3D convolutions used by convolution networks.
        ///
        struct conv3d_params_t
        {
                conv3d_params_t(
                        const tensor_size_t imaps = 0, const tensor_size_t irows = 0, const tensor_size_t icols = 0,
                        const tensor_size_t omaps = 1, const tensor_size_t kconn = 1,
                        const tensor_size_t krows = 1, const tensor_size_t kcols = 1,
                        const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1) :
                        m_imaps(imaps), m_irows(irows), m_icols(icols),
                        m_omaps(omaps), m_kconn(kconn),
                        m_krows(krows), m_kcols(kcols), m_kdrow(kdrow), m_kdcol(kdcol)
                {
                }

                conv3d_params_t(
                        const tensor3d_dim_t& idims,
                        const tensor_size_t omaps = 1, const tensor_size_t kconn = 1,
                        const tensor_size_t krows = 1, const tensor_size_t kcols = 1,
                        const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1) : conv3d_params_t(
                        std::get<0>(idims), std::get<1>(idims), std::get<2>(idims),
                        omaps, kconn, krows, kcols, kdrow, kdcol)
                {
                }

                auto imaps() const { return m_imaps; }
                auto irows() const { return m_irows; }
                auto icols() const { return m_icols; }
                auto idims() const { return make_dims(imaps(), irows(), icols()); }
                auto idims(const tensor_size_t count) const { return make_dims(count, imaps(), irows(), icols()); }
                auto isize() const { return nano::size(idims()); }

                auto krows() const { return m_krows; }
                auto kcols() const { return m_kcols; }
                auto kconn() const { return m_kconn; }
                auto kdrow() const { return m_kdrow; }
                auto kdcol() const { return m_kdcol; }

                auto omaps() const { return m_omaps; }
                auto orows() const { return (irows() - krows() + 1) / kdrow(); }
                auto ocols() const { return (icols() - kcols() + 1) / kdcol(); }
                auto odims() const { return make_dims(omaps(), orows(), ocols()); }
                auto odims(const tensor_size_t count) const { return make_dims(count, omaps(), orows(), ocols()); }
                auto osize() const { return nano::size(odims()); }

                auto kdims() const { return make_dims(omaps(), imaps() / kconn(), krows(), kcols()); }
                auto bdims() const { return omaps(); }

                auto psize() const { return imaps() * omaps() * krows() * kcols() / kconn() + omaps(); }

                auto flops_output() const { return 2 * imaps() * omaps() * orows() * ocols() * krows() * kcols() / kconn() + omaps() * orows() * ocols(); }
                auto flops_ginput() const { return 2 * imaps() * omaps() * irows() * icols() * krows() * kcols() / kconn(); }
                auto flops_gparam() const { return flops_output(); }

                auto make_idata(const tensor_size_t count) const { return tensor4d_t(idims(count)); }
                auto make_odata(const tensor_size_t count) const { return tensor4d_t(odims(count)); }
                auto make_kdata() const { return tensor4d_t(kdims()); }
                auto make_bdata() const { return vector_t(bdims()); }

                template <typename tidata, typename twdata, typename tbdata, typename todata>
                bool valid(const tidata&, const twdata&, const tbdata&, const todata&) const;

                bool valid_kernel() const { return kdrow() > 0 && kdcol() > 0 && orows() > 0 && ocols() > 0; }
                bool valid_connectivity() const { return (imaps() % kconn()) == 0 && (omaps() % kconn()) == 0; }
                bool valid() const { return valid_kernel() && valid_connectivity(); }

                // attributes
                tensor_size_t   m_imaps, m_irows, m_icols;      ///< input size
                tensor_size_t   m_omaps;                        ///< number of output planes
                tensor_size_t   m_kconn;                        ///< input-output plane connectivity factor
                tensor_size_t   m_krows, m_kcols;               ///< size of the convolution kernel
                tensor_size_t   m_kdrow, m_kdcol;               ///< convolution stride
        };

        template <typename tidata, typename tkdata, typename tbdata, typename todata>
        inline bool conv3d_params_t::valid(
                const tidata& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata) const
        {
                const auto count = idata.template size<0>();
                return  valid() &&
                        idata.template size<0>() == count &&
                        idata.template size<1>() == imaps() &&
                        idata.template size<2>() == irows() &&
                        idata.template size<3>() == icols() &&
                        kdata.dims() == kdims() &&
                        bdata.size() == bdims() &&
                        odata.template size<0>() == count &&
                        odata.template size<1>() == omaps() &&
                        odata.template size<2>() == orows() &&
                        odata.template size<3>() == ocols();
        }

        inline bool operator==(const conv3d_params_t& params1, const conv3d_params_t& params2)
        {
                return  params1.m_imaps == params2.m_imaps &&
                        params1.m_irows == params2.m_irows &&
                        params1.m_icols == params2.m_icols &&
                        params1.m_omaps == params2.m_omaps &&
                        params1.m_kconn == params2.m_kconn &&
                        params1.m_krows == params2.m_krows &&
                        params1.m_kcols == params2.m_kcols &&
                        params1.m_kdrow == params2.m_kdrow &&
                        params1.m_kdcol == params2.m_kdcol;
        }
}
