#pragma once

#include <cassert>
#include "vector.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (using a Toeplitz matrix)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv2d_toeplitz(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto osize = odata.size();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();
                        const auto isize = idata.size();

                        typename vector_types_t<tscalar>::tvector toeplitz_rowchunk(krows * icols - ocols + 1);
                        toeplitz_rowchunk.setZero();
                        for (auto kr = 0; kr < krows; kr ++)
                        {
                                toeplitz_rowchunk.segment(kr * icols, kcols) = kdata.row(kr);
                        }

                        tmatrixo toeplitz_matrix(osize, isize);
                        toeplitz_matrix.setZero();
                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto c = 0; c < ocols; c ++)
                                {
                                        toeplitz_matrix
                                        .row(r * ocols + c)
                                        .segment(r * icols + c, toeplitz_rowchunk.size()) = toeplitz_rowchunk;
                                }
                        }

                        tensor::map_vector(odata.data(), osize)
                                += toeplitz_matrix * tensor::map_vector(idata.data(), isize);
                }
        }
}

