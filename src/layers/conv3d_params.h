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
                        const tensor_size_t imaps = 1, const tensor_size_t irows = 1, const tensor_size_t icols = 1,
                        const tensor_size_t omaps = 1, const tensor_size_t kconn = 1,
                        const tensor_size_t krows = 1, const tensor_size_t kcols = 1,
                        const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1) :
                        m_imaps(imaps), m_irows(irows), m_icols(icols),
                        m_omaps(omaps), m_kconn(kconn),
                        m_krows(krows), m_kcols(kcols), m_kdrow(kdrow), m_kdcol(kdcol) {}

                auto imaps() const { return m_imaps; }
                auto irows() const { return m_irows; }
                auto icols() const { return m_icols; }
                auto idims() const { return tensor3d_dims_t{imaps(), irows(), icols()}; }
                auto isize() const { return nano::size(idims()); }

                auto krows() const { return m_krows; }
                auto kcols() const { return m_kcols; }
                auto kconn() const { return m_kconn; }
                auto kdrow() const { return m_kdrow; }
                auto kdcol() const { return m_kdcol; }

                auto omaps() const { return m_omaps; }
                auto orows() const { return (irows() - krows() + 1) / kdrow(); }
                auto ocols() const { return (icols() - kcols() + 1) / kdcol(); }
                auto odims() const { return tensor3d_dims_t{omaps(), orows(), ocols()}; }
                auto osize() const { return nano::size(odims()); }

                auto kdims() const { return tensor4d_dims_t{omaps(), imaps() / kconn(), krows(), kcols()}; }
                auto bdims() const { return omaps(); }

                auto psize() const { return imaps() * omaps() * krows() * kcols() / kconn() + omaps(); }
                auto flops_output() const { return 2 * imaps() * omaps() * orows() * ocols() * krows() * kcols() / kconn(); }
                auto flops_ginput() const { return 2 * imaps() * omaps() * irows() * icols() * krows() * kcols() / kconn(); }
                auto flops_gparam() const { return flops_output(); }

                auto make_idata() const { return tensor3d_t(idims()); }
                auto make_kdata() const { return tensor4d_t(kdims()); }
                auto make_bdata() const { return vector_t(bdims()); }
                auto make_odata() const { return tensor3d_t(odims()); }

                template <typename tidata>
                bool valid_idata(const tidata& idata) const { return idata.dims() == idims(); }

                template <typename tkdata>
                bool valid_kdata(const tkdata& kdata) const { return kdata.dims() == kdims(); }

                template <typename tbdata>
                bool valid_bdata(const tbdata& bdata) const { return bdata.size() == omaps(); }

                template <typename todata>
                bool valid_odata(const todata& odata) const { return odata.dims() == odims(); }

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
}
