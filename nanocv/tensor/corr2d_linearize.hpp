#pragma once

#include "matrix.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief create the Toeplitz-like matrix to replace
                ///     the 2D convolution: idata += odata @ kdata
                ///     with the linear product: as_vector(idata) = toeplitz * as_vector(kdata)
                ///
                template
                <
                        typename tmatrixo,
                        typename tsize
                >
                decltype(auto) corr2d_linearize(const tmatrixo& odata, const tsize krows, const tsize kcols)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto irows = orows + krows - 1;
                        const auto icols = ocols + kcols - 1;
                        const auto isize = irows * icols;
                        const auto ksize = krows * kcols;

                        typedef typename tmatrixo::Scalar                               tscalar;
                        typedef typename tensor::matrix_types_t<tscalar>::tmatrix       ttoeplitz;

                        ttoeplitz toeplitz = ttoeplitz::Zero(ksize, isize);

                        /// \todo more efficient construction
                        for (tsize r = 0; r < orows; r ++)
                        {
                                for (tsize kr = 0; kr < krows; kr ++)
                                {
                                        for (tsize c = 0; c < ocols; c ++)
                                        {
                                                for (tsize kc = 0; kc < kcols; kc ++)
                                                {
                                                        toeplitz(kr * kcols + kc, (r + kr) * icols + (c + kc)) +=
                                                        odata(r, c);
                                                }
                                        }
                                }
                        }

                        return toeplitz;
                }

                template
                <
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo
                >
                decltype(auto) corr2d_linearize(const tmatrixo& odata, const tmatrixk& kdata)
                {
                        return corr2d_linearize(odata, kdata.rows(), kdata.cols());
                }
        }
}

