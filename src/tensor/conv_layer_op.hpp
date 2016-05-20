#pragma once

#include "conv2d_dyn.hpp"
#include "corr2d_dyn.hpp"
#include "conv_layer_params.hpp"

namespace tensor
{
        template <typename tsize>
        struct conv_layer_param_t
        {
                explicit conv_layer_param_t(
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

                        // todo: assert kdata
                }

                tsize           m_idims, m_irows, m_icols;
                tsize           m_odims, m_orows, m_ocols;
                tsize           m_krows, m_kcols, m_kconn;
        };

        template <typename tsize>
        class conv_layer_op_t
        {
        public:

                explicit conv_layer_op_t(
                        const tsize idims, const tsize irows, const tsize icols,
                        const tsize odims, const tsize krows, const tsize kcols, const tsize kconn) :
                        m_params(idims, irows, icols, odims, krows, kcols, kconn)
                {
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void output(const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
                {
                        m_params.check(idata, kdata, odata);

                        odata.setZero();

                        conv2d_dyn_t op;
                        conv_loop([&] (const auto i, const auto ik, const auto o)
                        {
                                op(idata.matrix(i), kdata.matrix(o, ik), odata.matrix(o));
                        });
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
                {
                        m_params.check(idata, kdata, odata);

                        kdata.setZero();

                        conv2d_dyn_t op;
                        conv_loop([&] (const auto i, const auto ik, const auto o)
                        {
                                op(idata.matrix(i), odata.matrix(o), kdata.matrix(o, ik));
                        });
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void gpinput(ttensori&& idata, const ttensork& kdata, const ttensoro& odata)
                {
                        m_params.check(idata, kdata, odata);

                        idata.setZero();

                        corr2d_dyn_t op;
                        conv_loop([&] (const auto i, const auto ik, const auto o)
                        {
                                op(odata.matrix(o), kdata.matrix(o, ik), idata.matrix(i));
                        });
                }

        private:

                template <typename toperator>
                void conv_loop(const toperator& op)
                {
                        const auto kconn = m_params.m_kconn;
                        for (tsize o = 0; o < m_params.m_odims; ++ o)
                        {
                                for (tsize i = (o % kconn), ik = 0; i < m_params.m_idims; ++ ik, i += kconn)
                                {
                                        op(i, ik, o);
                                }
                        }
                }

                conv_layer_param_t<tsize>       m_params;
        };
}


