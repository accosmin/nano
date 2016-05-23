#pragma once

#include "tensor/conv2d_dyn.hpp"
#include "tensor/corr2d_dyn.hpp"
#include "convolution_params.h"

namespace nano
{
        template
        <
                typename tconv2d = tensor::conv2d_dyn_t,        ///< operator to compute 2D convolutions
                typename tcorr2d = tensor::corr2d_dyn_t         ///< operator to compute 2D correlations
        >
        class convolution_kernel2d_t
        {
        public:

                convolution_kernel2d_t(
                        const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t odims, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn) :
                        m_params(idims, irows, icols, odims, krows, kcols, kconn),
                        m_convop(tconv2d()),
                        m_corrop(tcorr2d())
                {
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void output(const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
                {
                        m_params.check(idata, kdata, odata);

                        odata.setZero();
                        conv_loop([&] (const auto i, const auto ik, const auto o)
                        {
                                m_convop(idata.matrix(i), kdata.matrix(o, ik), odata.matrix(o));
                        });
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
                {
                        m_params.check(idata, kdata, odata);

                        kdata.setZero();
                        conv_loop([&] (const auto i, const auto ik, const auto o)
                        {
                                m_convop(idata.matrix(i), odata.matrix(o), kdata.matrix(o, ik));
                        });
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void ginput(ttensori&& idata, const ttensork& kdata, const ttensoro& odata)
                {
                        m_params.check(idata, kdata, odata);

                        idata.setZero();
                        conv_loop([&] (const auto i, const auto ik, const auto o)
                        {
                                m_corrop(odata.matrix(o), kdata.matrix(o, ik), idata.matrix(i));
                        });
                }

        private:

                tensor_size_t idims() const { return m_params.m_idims; }
                tensor_size_t odims() const { return m_params.m_odims; }
                tensor_size_t kconn() const { return m_params.m_kconn; }

                template <typename toperator>
                void conv_loop(const toperator& op)
                {
                        for (tensor_size_t o = 0; o < odims(); ++ o)
                        {
                                for (tensor_size_t i = (o % kconn()), ik = 0; i < idims(); ++ ik, i += kconn())
                                {
                                        op(i, ik, o);
                                }
                        }
                }

                // attributes
                convolution_params_t    m_params;
                tconv2d                 m_convop;
                tcorr2d                 m_corrop;
        };
}


