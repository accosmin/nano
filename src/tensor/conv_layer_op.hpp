#pragma once

#include "conv2d_dyn.hpp"
#include "corr2d_dyn.hpp"
#include "conv_layer_params.hpp"

namespace tensor
{
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

                conv_layer_params_t<tsize>      m_params;
        };
}


