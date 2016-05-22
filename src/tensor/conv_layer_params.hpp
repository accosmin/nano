#pragma once

#include <cassert>

namespace tensor
{
        template <typename tsize>
        struct conv_layer_params_t
        {
                conv_layer_params_t(
                        const tsize idims, const tsize irows, const tsize icols,
                        const tsize odims, const tsize krows, const tsize kcols, const tsize kconn) :
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

                        assert(idata.size<0>() == m_idims);
                        assert(idata.size<1>() == m_irows);
                        assert(idata.size<2>() == m_icols);

                        assert(odata.size<0>() == m_odims);
                        assert(odata.size<1>() == m_orows);
                        assert(odata.size<2>() == m_ocols);

                        assert(kdata.size<0>() == m_odims);
                        assert(kdata.size<1>() == m_idims / m_kconn);
                        assert(kdata.size<2>() == m_krows);
                        assert(kdata.size<3>() == m_kcols);
                }

                tsize           m_idims, m_irows, m_icols;
                tsize           m_odims, m_orows, m_ocols;
                tsize           m_krows, m_kcols, m_kconn;
        };
}


