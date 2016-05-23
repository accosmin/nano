#pragma once

#include "convolution_params.h"

namespace nano
{
        class convolution_toeplitz_t
        {
        public:

                convolution_toeplitz_t(
                        const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t odims, const tensor_size_t krows, const tensor_size_t kcols, const tensor_size_t kconn) :
                        m_params(idims, irows, icols, odims, krows, kcols, kconn),
                        m_ikdata(krows * kcols, orows() * ocols()),
                        m_iodata(orows() * ocols(), krows * kcols),
                        m_okdata(krows * kcols, irows * icols)
                {
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void output(const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
                {
                        m_params.check(idata, kdata, odata);

                        // todo: assuming kdata(i, o)

                        odata.setZero();
                        for (tensor_size_t i = 0; i < idims(); ++ i)
                        {
                                make_conv(idata.matrix(i), krows(), kcols(), m_ikdata);

                                tensor::map_matrix(odata.data(), odims(), orows() * ocols()) +=
                                        tensor::map_matrix(kdata.planeData(i, 0), odims(), krows() * kcols()) *
                                        m_ikdata;
                        }
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
                {
                        m_params.check(idata, kdata, odata);

                        // todo: assuming kdata(i, o)

                        kdata.setZero();
                        for (tensor_size_t i = 0; i < idims(); ++ i)
                        {
                                make_conv(idata.matrix(i), orows(), ocols(), m_iodata);

                                tensor::map_matrix(kdata.planeData(i, 0), odims(), krows() * kcols()) =
                                        tensor::map_matrix(odata.data(), odims(), orows() * ocols()) *
                                        m_iodata;
                        }
                }

                template <typename ttensori, typename ttensork, typename ttensoro>
                void ginput(ttensori&& idata, const ttensork& kdata, const ttensoro& odata)
                {
                        m_params.check(idata, kdata, odata);

                        // todo: assuming kdata(o, i)

                        idata.setZero();
                        for (tensor_size_t o = 0; o < idims(); ++ o)
                        {
                                make_corr(odata.matrix(o), krows(), kcols(), m_okdata);

                                tensor::map_matrix(idata.data(), idims(), irows() * icols()) +=
                                        tensor::map_matrix(kdata.planeData(o, 0), idims(), krows() * kcols()) *
                                        m_okdata;
                        }
                }

        private:

                tensor_size_t idims() const { return m_params.m_idims; }
                tensor_size_t irows() const { return m_params.m_irows; }
                tensor_size_t icols() const { return m_params.m_icols; }
                tensor_size_t odims() const { return m_params.m_odims; }
                tensor_size_t orows() const { return m_params.m_orows; }
                tensor_size_t ocols() const { return m_params.m_ocols; }
                tensor_size_t kconn() const { return m_params.m_kconn; }
                tensor_size_t krows() const { return m_params.m_krows; }
                tensor_size_t kcols() const { return m_params.m_kcols; }

                template <typename timatrix, typename tensor_size_t, typename tomatrix>
                static void make_conv(const timatrix& imat, const tensor_size_t krows, const tensor_size_t kcols, tomatrix& omat)
                {
                        const tensor_size_t irows = imat.rows();
                        const tensor_size_t icols = imat.cols();
                        const tensor_size_t orows = irows - krows + 1;
                        const tensor_size_t ocols = icols - kcols + 1;

                        assert(omat.rows() == krows * kcols);
                        assert(omat.cols() == orows * ocols);

                        for (tensor_size_t kr = 0; kr < krows; ++ kr)
                        {
                                for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                {
                                        for (tensor_size_t r = 0; r < orows; ++ r)
                                        {
                                                for (tensor_size_t c = 0; c < ocols; ++ c)
                                                {
                                                        omat(kr * kcols + kc, r * ocols + c) = imat(r + kr, c + kc);
                                                }
                                        }
                                }
                        }
                }

                template <typename tomatrix, typename tensor_size_t, typename timatrix>
                static void make_corr(const tomatrix& omat, const tensor_size_t krows, const tensor_size_t kcols, timatrix& imat)
                {
                        const tensor_size_t orows = omat.rows();
                        const tensor_size_t ocols = omat.cols();
                        const tensor_size_t irows = orows + krows - 1;
                        const tensor_size_t icols = ocols + kcols - 1;

                        NANO_UNUSED1_RELEASE(irows);

                        assert(imat.rows() == krows * kcols);
                        assert(imat.cols() == irows * icols);

                        imat.setZero();
                        for (tensor_size_t kr = 0; kr < krows; ++ kr)
                        {
                                for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                {
                                        for (tensor_size_t r = 0; r < orows; ++ r)
                                        {
                                                for (tensor_size_t c = 0; c < ocols; ++ c)
                                                {
                                                        imat(kr * kcols + kc, (r + kr) * icols + c + kc) += omat(r, c);
                                                }
                                        }
                                }
                        }
                }

                // attributes
                convolution_params_t    m_params;
                matrix_t                m_ikdata;       ///< input-kernel convolution
                matrix_t                m_iodata;       ///< input-output convolution
                matrix_t                m_okdata;       ///< output-kernel correlation
        };
}

