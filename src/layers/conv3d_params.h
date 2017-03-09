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
                        const tensor_size_t imaps, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t omaps, const tensor_size_t kconn,
                        const tensor_size_t krows, const tensor_size_t kcols,
                        const tensor_size_t kdrow, const tensor_size_t kdcol) :
                        m_imaps(imaps), m_irows(irows), m_icols(icols),
                        m_omaps(omaps), m_kconn(kconn),
                        m_krows(krows), m_kcols(kcols), m_kdrow(kdrow), m_kdcol(kdcol) {}

                tensor_size_t imaps() const { return m_imaps; }
                tensor_size_t irows() const { return m_irows; }
                tensor_size_t icols() const { return m_icols; }

                tensor_size_t krows() const { return m_krows; }
                tensor_size_t kcols() const { return m_kcols; }
                tensor_size_t kconn() const { return m_kconn; }
                tensor_size_t kdrow() const { return m_kdrow; }
                tensor_size_t kdcol() const { return m_kdcol; }

                tensor_size_t omaps() const { return m_omaps; }
                tensor_size_t orows() const { return (irows() - krows() + 1) / kdrow(); }
                tensor_size_t ocols() const { return (irows() - krows() + 1) / kdcol(); }

                tensor_size_t psize() const { return imaps() * omaps() * krows() * kcols() / kconn() + omaps(); }
                tensor_size_t flops() const { return 2 * imaps() * omaps() * orows() * ocols() * krows() * kcols() / kconn(); }

                operator bool() const { return orows() > 0 && ocols() > 0 && kconn() <= imaps(); }

                tensor_size_t   m_imaps, m_irows, m_icols;      ///< input size
                tensor_size_t   m_omaps;                        ///< number of output planes
                tensor_size_t   m_kconn;                        ///< input-output plane connectivity factor
                tensor_size_t   m_krows, m_kcols;               ///< size of the convolution kernel
                tensor_size_t   m_kdrow, m_kdcol;               ///< convolution stride
        };
}
