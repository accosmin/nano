#pragma once

#include <cassert>
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
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi
                >
                decltype(auto) make_toeplitz(const tmatrixi& idata, const tmatrixk& kdata, const tmatrixo& odata)
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto osize = odata.size();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto ksize = kdata.size();

                        typedef typename tmatrixi::Scalar                               tscalar;
                        typedef typename tensor::matrix_types_t<tscalar>::tmatrix       ttoeplitz;

                        ttoeplitz toeplitz = ttoeplitz::Zero(osize, ksize);

                        /// \todo more efficient construction
                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        for (auto c = 0; c < ocols; c ++)
                                        {
//                                                toeplitz.row(r * ocols + c).segment(kr * kcols, krows) =
//                                                idata.row(r + kr).segment(c, kcols);

                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        toeplitz(r * ocols + c, kr * kcols + kc) =
                                                        idata(r + kr, c + kc);
                                                }
                                        }
                                }
                        }

                        return toeplitz;
                }
        }
}

