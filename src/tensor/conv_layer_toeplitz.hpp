#pragma once

#include "conv_layer_params.hpp"

namespace nano
{
        template <typename tsize>
        class conv_layer_toeplitz_t
        {
        public:

                explicit conv_layer_toeplitz_t(
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

                        for (tsize i = 0; i < m_params.m_idims; ++ i)
                        {
                                make_conv(idata.matrix(i), m_params.m_krows, m_params.kcols, m_toeiodata);

                                const auto kdata = tensor::map_matrix(m_kdata.planeData(i, 0), odims(), krows() * kcols());
                                tensor::map_matrix(m_odata.data(), odims(), orows() * ocols()) += kdata * m_toeiodata;
                        }
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
                {
                        m_params.check(idata, kdata, odata);

                        kdata.setZero();

                        for (tensor_size_t i = 0; i < idims(); ++ i)
                        {
                                make_conv(m_idata.matrix(i), orows(), ocols(), m_toeikdata);

                                const auto odata = tensor::map_matrix(m_odata.data(), odims(), orows() * ocols());
                                tensor::map_matrix(gkdata.planeData(i, 0), odims(), krows() * kcols()) = odata * m_toeikdata;
                        }
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void gpinput(ttensori&& idata, const ttensork& kdata, const ttensoro& odata)
                {
                        m_params.check(idata, kdata, odata);

                        idata.setZero();

                        for (tensor_size_t o = 0; o < odims(); ++ o)
                        {
                                make_corr(m_odata.matrix(o), krows(), kcols(), m_toeokdata);

                                for (tensor_size_t i = 0; i < idims(); ++ i)
                                {
                                        m_toekdata.row(i) = m_kdata.vector(i, o);
                                }

                                //const auto kdata = tensor::map_matrix(m_kdata.planeData(o, 0), idims(), krows() * kcols());
                                tensor::map_matrix(m_idata.data(), idims(), irows() * icols()) += m_toekdata * m_toeokdata;
                        }
                }

        private:

                template <typename timatrix, typename tsize, typename tomatrix>
                static void make_conv(const timatrix& imat, const tsize krows, const tsize kcols, tomatrix& omat)
                {
                        const tsize irows = imat.rows();
                        const tsize icols = imat.cols();
                        const tsize orows = irows - krows + 1;
                        const tsize ocols = icols - kcols + 1;

                        assert(omat.rows() == krows * kcols);
                        assert(omat.cols() == orows * ocols);

                        for (tsize kr = 0; kr < krows; ++ kr)
                        {
                                for (tsize kc = 0; kc < kcols; ++ kc)
                                {
                                        for (tsize r = 0; r < orows; ++ r)
                                        {
                                                for (tsize c = 0; c < ocols; ++ c)
                                                {
                                                        omat(kr * kcols + kc, r * ocols + c) = imat(r + kr, c + kc);
                                                }
                                        }
                                }
                        }
                }

                template <typename tomatrix, typename tsize, typename timatrix>
                static void make_corr(const tomatrix& omat, const tsize krows, const tsize kcols, timatrix& imat)
                {
                        const tsize orows = omat.rows();
                        const tsize ocols = omat.cols();
                        const tsize irows = orows + krows - 1;
                        const tsize icols = ocols + kcols - 1;

                        NANO_UNUSED1_RELEASE(irows);

                        assert(imat.rows() == krows * kcols);
                        assert(imat.cols() == irows * icols);

                        imat.setZero();
                        for (tsize kr = 0; kr < krows; ++ kr)
                        {
                                for (tsize kc = 0; kc < kcols; ++ kc)
                                {
                                        for (tsize r = 0; r < orows; ++ r)
                                        {
                                                for (tsize c = 0; c < ocols; ++ c)
                                                {
                                                        imat(kr * kcols + kc, (r + kr) * icols + c + kc) += omat(r, c);
                                                }
                                        }
                                }
                        }
                }

                conv_layer_params_t<tsize>      m_params;
        };
}


