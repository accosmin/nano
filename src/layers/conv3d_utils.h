#pragma once

#include <cassert>
#include "conv3d_params.h"

namespace nano
{
        template <typename timatrix, typename tomatrix>
        void img2col(const conv3d_params_t& params, const timatrix& imat, tomatrix&& omat)
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
}
