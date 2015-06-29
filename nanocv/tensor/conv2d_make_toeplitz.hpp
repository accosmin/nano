#pragma once

#include "matrix.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief create the Toeplitz-like matrix to replace
                ///     the 2D convolution: odata += idata @ kdata
                ///     with the linear product: as_vector(odata) = toeplitz * as_vector(kdata)
                ///
                template
                <
                        typename tmatrixi,
                        typename tsize
                >
                decltype(auto) make_toeplitz(const tmatrixi& idata, const tsize krows, const tsize kcols)
                {
                        const auto orows = idata.rows() - krows + 1;
                        const auto ocols = idata.cols() - kcols + 1;
                        const auto osize = orows * ocols;
                        const auto ksize = krows * kcols;

                        typedef typename tmatrixi::Scalar                               tscalar;
                        typedef typename tensor::matrix_types_t<tscalar>::tmatrix       ttoeplitz;

                        ttoeplitz toeplitz = ttoeplitz::Zero(osize, ksize);

                        /// \todo more efficient construction
                        for (tsize r = 0; r < orows; r ++)
                        {
                                for (tsize kr = 0; kr < krows; kr ++)
                                {
                                        for (tsize c = 0; c < ocols; c ++)
                                        {
//                                                toeplitz.row(r * ocols + c).segment(kr * kcols, krows) =
//                                                idata.row(r + kr).segment(c, kcols);

                                                for (tsize kc = 0; kc < kcols; kc ++)
                                                {
                                                        toeplitz(r * ocols + c, kr * kcols + kc) =
                                                        idata(r + kr, c + kc);
                                                }
                                        }
                                }
                        }

                        return toeplitz;
                }

                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi
                >
                decltype(auto) make_toeplitz(const tmatrixi& idata, const tmatrixk& kdata)
                {
                        return make_toeplitz(idata, kdata.rows(), kdata.cols());
                }
        }
}

