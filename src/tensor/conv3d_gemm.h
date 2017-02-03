#pragma once

#include "tensor.h"
#include <cassert>

namespace tensor
{
        template <typename tscalar>
        struct conv3d_traits_t
        {
                using ttensor3d = tensor_t<tscalar, 3>;
                using ttensor4d = tensor_t<tscalar, 4>;

                using tdims3d = typename ttensor3d::tdims;
                using tdims4d = typename ttensor4d::tdims;

                using tindex = typename ttensor3d::Index;
        };

        template <typename tscalar>
        struct conv3d_gemm_t
        {
                tdims3d         m_isize;        ///< input size: idims x irows x icols
                ttensor4d       m_kdata;        ///< convolution kernels: odims x idims/kconn x krows x kcols
                tindex          m_kconn;        ///< input-output connectivity factory
                tindex          m_kdrow;        ///< convolution stride (rows)
                tindex          m_kdcol;        ///< convolution stride (cols)
                tdims3d         m_osize;        ///< output size: odims x orows x ocols
        };

        ///
        /// \brief
        ///
        template <typename timatrix, typename tsize, typename tomatrix>
        void make_toeplitz_output(const timatrix& imat,
                const tsize orows, const tsize ocols,
                const tsize krows, const tsize kcols,
                const tsize drows, const tsize dcols,
                tomatrix&& omat)
        {
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
                                                omat(kr * kcols + kc, r * ocols + c) =
                                                imat(r * drows + kr, c * dcols + kc);
                                        }
                                }
                        }
                }
        }

        ///
        /// \brief
        ///
        template <typename timatrix, typename tsize, typename tkmatrix>
        void make_toeplitz_gparam(const timatrix& imat,
                const tsize orows, const tsize ocols,
                const tsize krows, const tsize kcols,
                const tsize drows, const tsize dcols,
                tkmatrix&& kmat)
        {
                assert(kmat.rows() == orows * ocols);
                assert(kmat.cols() == krows * kcols);

                for (tsize r = 0; r < orows; ++ r)
                {
                        for (tsize c = 0; c < ocols; ++ c)
                        {
                                for (tsize kr = 0; kr < krows; ++ kr)
                                {
                                        for (tsize kc = 0; kc < kcols; ++ kc)
                                        {
                                                kmat(r * ocols + c, kr * kcols + kc) =
                                                imat(r * drows + kr, c * dcols + kc);
                                        }
                                }
                        }
                }
        }

        ///
        /// \brief
        ///
        template <typename tomatrix, typename tsize, typename timatrix>
        void make_toeplitz_ginput(const tomatrix& omat,
                const tsize orows, const tsize ocols,
                const tsize krows, const tsize kcols,
                const tsize drows, const tsize dcols,
                const tsize irows, const tsize icols,
                timatrix& imat)
        {
                assert(imat.rows() == krows * kcols);
                assert(imat.cols() == irows * icols);

                NANO_UNUSED1_RELEASE(irows);

                imat.setZero();
                for (tsize kr = 0; kr < krows; ++ kr)
                {
                        for (tsize kc = 0; kc < kcols; ++ kc)
                        {
                                for (tsize r = 0; r < orows; ++ r)
                                {
                                        for (tsize c = 0; c < ocols; ++ c)
                                        {
                                                imat(kr * kcols + kc, (r * drows + kr) * icols + c * dcols + kc) +=
                                                omat(r, c);
                                        }
                                }
                        }
                }
        }
}
