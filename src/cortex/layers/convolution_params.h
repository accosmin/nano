#pragma once

#include "tensor.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief describe a convolution layer:
        ///     input tensor: idims x irows x icols
        ///     convolution kernel: odims x idims/kconn x krows x kcols
        ///     output tensor: odims x orows x ocols
        ///
        struct convolution_params_t
        {
                convolution_params_t(
                        const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t odims, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn) :
                        m_idims(idims), m_irows(irows), m_icols(icols),
                        m_odims(odims), m_orows(irows - krows + 1), m_ocols(icols - kcols + 1),
                        m_krows(krows), m_kcols(kcols), m_kconn(kconn)
                {
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void check(const ttensori& idata, const ttensork& kdata, const ttensoro& odata)
                {
                        assert(idata.dimensionality() == 3);
                        assert(kdata.dimensionality() == 4);
                        assert(odata.dimensionality() == 3);

                        assert(idata.template size<0>() == m_idims);
                        assert(idata.template size<1>() == m_irows);
                        assert(idata.template size<2>() == m_icols);

                        assert(odata.template size<0>() == m_odims);
                        assert(odata.template size<1>() == m_orows);
                        assert(odata.template size<2>() == m_ocols);

                        assert(kdata.template size<0>() == m_odims);
                        assert(kdata.template size<1>() == m_idims / m_kconn);
                        assert(kdata.template size<2>() == m_krows);
                        assert(kdata.template size<3>() == m_kcols);
                }

                tensor_size_t   m_idims, m_irows, m_icols;
                tensor_size_t   m_odims, m_orows, m_ocols;
                tensor_size_t   m_krows, m_kcols, m_kconn;
        };
}


