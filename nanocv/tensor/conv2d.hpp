#pragma once

#include "matrix.hpp"
#include <cassert>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief create the Toeplitz-like matrix to replace
                ///     the 2D convolution: odata += idata @ kdata
                ///     with the more efficient: as_vector(odata) = transpose(ldata) * as_vector(kdata)
                ///
                template
                <
                        typename tmatrixi,
                        typename tsize,
                        typename tmatrixl               ///< linearized matrix (asssuming already allocated)
                >
                void linearize_conv2d(const tmatrixi& idata, const tsize krows, const tsize kcols, tmatrixl&& ldata)
                {
                        const tsize orows = idata.rows() - krows + 1;
                        const tsize ocols = idata.cols() - kcols + 1;

                        assert(ldata.rows() == krows * kcols);
                        assert(ldata.cols() == orows * ocols);

                        /// \todo more efficient construction
                        for (tsize r = 0; r < orows; r ++)
                        {
                                for (tsize kr = 0; kr < krows; kr ++)
                                {
                                        for (tsize c = 0; c < ocols; c ++)
                                        {
                                                for (tsize kc = 0; kc < kcols; kc ++)
                                                {
                                                        ldata(kr * kcols + kc, r * ocols + c) =
                                                        idata(r + kr, c + kc);
                                                }
                                        }
                                }
                        }
                }
        }
}

