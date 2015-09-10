#pragma once

#include "matrix.hpp"
#include <cassert>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief create the Toeplitz-like matrix to replace
                ///     the 2D convolution: idata += odata @ kdata
                ///     with the more eficient: as_vector(idata) = transpose(ldata) * as_vector(kdata)
                ///
                 template
                <
                        typename tmatrixo,
                        typename tsize,
                        typename tmatrixl               ///< linearized matrix (asssuming already allocated)
                >
                void linearize_corr2d(const tmatrixo& odata, const tsize krows, const tsize kcols, tmatrixl&& ldata)
                {
                        const tsize orows = odata.rows();
                        const tsize ocols = odata.cols();
                        const tsize icols = ocols + kcols - 1;

                        assert(ldata.rows() == krows * kcols);
                        assert(ldata.cols() == (orows + krows - 1) * icols);

                        ldata.setZero();

                        /// \todo more efficient construction
                        for (tsize kr = 0; kr < krows; kr ++)
                        {
                                for (tsize kc = 0; kc < kcols; kc ++)
                                {
                                        for (tsize r = 0; r < orows; r ++)
                                        {
                                                for (tsize c = 0; c < ocols; c ++)
                                                {
                                                        ldata(kr * kcols + kc, (r + kr) * icols + (c + kc)) +=
                                                        odata(r, c);
                                                }
                                        }
                                }
                        }
                }
        }
}

