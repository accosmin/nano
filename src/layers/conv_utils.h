#pragma once

#include <cassert>
#include "tensor.h"

namespace nano
{
        template <typename timatrix, typename tomatrix>
        void img2col0(const timatrix& imat,
                const tensor_size_t orows, const tensor_size_t ocols,
                const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t drows, const tensor_size_t dcols,
                tomatrix&& omat)
        {
                assert(orows == (imat.rows() - krows + 1) / drows);
                assert(ocols == (imat.cols() - kcols + 1) / dcols);
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

        template <typename timatrix, typename tomatrix>
        void img2colx(const timatrix& imat,
                const tensor_size_t orows, const tensor_size_t ocols,
                const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t drows, const tensor_size_t dcols,
                tomatrix&& omat)
        {
                assert(orows == (imat.rows() - krows + 1) / drows);
                assert(ocols == (imat.cols() - kcols + 1) / dcols);
                assert(omat.rows() == krows * kcols);
                assert(omat.cols() == orows * ocols);

                if (drows == 1 && dcols == 1)
                {
                        for (tensor_size_t kr = 0; kr < krows; ++ kr)
                        {
                                for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                {
                                        for (tensor_size_t r = 0; r < orows; ++ r)
                                        {
                                                for (tensor_size_t c = 0; c < ocols; ++ c)
                                                {
                                                        omat(kr * kcols + kc, r * ocols + c) =
                                                        imat(r + kr, c + kc);
                                                }
                                        }
                                }
                        }
                }
                else
                {
                        img2col0(imat, orows, ocols, krows, kcols, drows, dcols, omat);
                }
        }

        template <typename timatrix, typename tomatrix>
        void col2img(timatrix&& imat,
                const tensor_size_t orows, const tensor_size_t ocols,
                const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t drows, const tensor_size_t dcols,
                const tomatrix& omat)
        {
                assert(orows == (imat.rows() - krows + 1) / drows);
                assert(ocols == (imat.cols() - kcols + 1) / dcols);
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
                                                imat(r * drows + kr, c * dcols + kc) +=
                                                omat(kr * kcols + kc, r * ocols + c);
                                        }
                                }
                        }
                }
        }

        template <typename timatrix, typename tkmatrix, typename tomatrix>
        void convo2d(const timatrix& imat, const tkmatrix& kmat, const tensor_size_t drows, const tensor_size_t dcols,
                tomatrix&& omat)
        {
                const auto krows = kmat.rows(), kcols = kmat.cols();
                const auto orows = omat.rows(), ocols = omat.cols();

                assert(orows == (imat.rows() - krows + 1) / drows);
                assert(ocols == (imat.cols() - kcols + 1) / dcols);

                for (tensor_size_t r = 0; r < orows; ++ r)
                {
                        for (tensor_size_t c = 0; c < ocols; ++ c)
                        {
                                for (tensor_size_t kr = 0; kr < krows; ++ kr)
                                {
                                        for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                        {
                                                omat(r, c) += imat(r * drows + kr, c * dcols + kc) * kmat(kr, kc);
                                        }
                                }
                        }
                }
        }

        template <typename timatrix, typename tkmatrix, typename tomatrix>
        void convk2d(const timatrix& imat, tkmatrix&& kmat, const tensor_size_t drows, const tensor_size_t dcols,
                const tomatrix& omat)
        {
                const auto krows = kmat.rows(), kcols = kmat.cols();
                const auto orows = omat.rows(), ocols = omat.cols();

                assert(orows == (imat.rows() - krows + 1) / drows);
                assert(ocols == (imat.cols() - kcols + 1) / dcols);

                for (tensor_size_t r = 0; r < orows; ++ r)
                {
                        for (tensor_size_t c = 0; c < ocols; ++ c)
                        {
                                for (tensor_size_t kr = 0; kr < krows; ++ kr)
                                {
                                        for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                        {
                                                kmat(kr, kc) += imat(r * drows + kr, c * dcols + kc) * omat(r, c);
                                        }
                                }
                        }
                }
        }

        template <typename timatrix, typename tkmatrix, typename tomatrix>
        void convi2d(timatrix&& imat, const tkmatrix& kmat, const tensor_size_t drows, const tensor_size_t dcols,
                const tomatrix& omat)
        {
                const auto krows = kmat.rows(), kcols = kmat.cols();
                const auto orows = omat.rows(), ocols = omat.cols();

                assert(orows == (imat.rows() - krows + 1) / drows);
                assert(ocols == (imat.cols() - kcols + 1) / dcols);

                for (tensor_size_t r = 0; r < orows; ++ r)
                {
                        for (tensor_size_t c = 0; c < ocols; ++ c)
                        {
                                for (tensor_size_t kr = 0; kr < krows; ++ kr)
                                {
                                        for (tensor_size_t kc = 0; kc < kcols; ++ kc)
                                        {
                                                imat(r * drows + kr, c * dcols + kc) += omat(r, c) * kmat(kr, kc);
                                        }
                                }
                        }
                }
        }
}
