#pragma once

#include <cassert>
#include "conv_params.h"

namespace nano
{
        template <typename timatrix, typename tomatrix>
        void img2col(const conv_params_t& params, const timatrix& imat, tomatrix&& omat)
        {
                const auto orows = params.orows(), ocols = params.ocols();
                const auto krows = params.krows(), kcols = params.kcols();
                const auto drows = params.kdrow(), dcols = params.kdcol();

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
                                                omat(kr * kcols + kc, r * ocols + c) =
                                                imat(r * drows + kr, c * dcols + kc);
                                        }
                                }
                        }
                }
        }

        template <typename timatrix, typename tkmatrix, typename tomatrix>
        void conv2d(const timatrix& imat, const tkmatrix& kmat, const tensor_size_t dr, const tensor_size_t dc,
                tomatrix&& omat)
        {
                for (tensor_size_t orows = omat.rows(), r = 0; r < orows; ++ r)
                {
                        for (tensor_size_t ocols = omat.cols(), c = 0; c < ocols; ++ c)
                        {
                                for (tensor_size_t krows = kmat.rows(), kr = 0; kr < krows; ++ kr)
                                {
                                        for (tensor_size_t kcols = kmat.cols(), kc = 0; kc < kcols; ++ kc)
                                        {
                                                omat(r, c) += imat(r * dr + kr, c * dc + kc) * kmat(kr, kc);
                                        }
                                }
                        }
                }
        }

        template <typename timatrix, typename tkmatrix, typename tomatrix>
        void corr2d(timatrix&& imat, const tkmatrix& kmat, const tensor_size_t dr, const tensor_size_t dc, const tomatrix& omat)
        {
                for (tensor_size_t orows = omat.rows(), r = 0; r < orows; ++ r)
                {
                        for (tensor_size_t ocols = omat.cols(), c = 0; c < ocols; ++ c)
                        {
                                for (tensor_size_t krows = kmat.rows(), kr = 0; kr < krows; ++ kr)
                                {
                                        for (tensor_size_t kcols = kmat.cols(), kc = 0; kc < kcols; ++ kc)
                                        {
                                                imat(r * dr + kr, c * dc + kc) += omat(r, c) * kmat(kr, kc);
                                        }
                                }
                        }
                }
        };
}
