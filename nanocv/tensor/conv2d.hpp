#pragma once

#include <cassert>
#include "vector.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief create a Toeplitz matrix to be used for the 2D convolution: odata += idata @ kdata
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                tmatrixo conv2d_make_toeplitz(const tmatrixi& idata, const tmatrixk& kdata, const tmatrixo& odata)
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

                        typedef typename vector_types_t<tscalar>::tvector tvector;

                        tvector toeplitz_rowchunk = tvector::Zero(krows * icols - ocols + 1);
                        for (auto kr = 0; kr < krows; kr ++)
                        {
                                toeplitz_rowchunk.segment(kr * icols, kcols) = kdata.row(kr);
                        }

                        tmatrixo toeplitz_matrix = tmatrixo::Zero(osize, isize);
                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto c = 0; c < ocols; c ++)
                                {
                                        toeplitz_matrix
                                        .row(r * ocols + c)
                                        .segment(r * icols + c, toeplitz_rowchunk.size()) = toeplitz_rowchunk;
                                }
                        }

                        return toeplitz_matrix;
                }

                ///
                /// \brief 2D convolution: odata += idata @ kdata (using an already computed Toeplitz matrix)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv2d_toeplitz_buffered(
                        const tmatrixi& idata, const tmatrixk& kdata, const tmatrixo& toeplitz, tmatrixo& odata)
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto osize = odata.size();
                        const auto isize = idata.size();

                        tensor::map_vector(odata.data(), osize) += toeplitz * tensor::map_vector(idata.data(), isize);
                }

                ///
                /// \brief 2D convolution: odata += idata @ kdata (using a Toeplitz matrix computed on the fly)
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

                        return conv2d_toeplitz_buffered(idata, kdata, conv2d_make_toeplitz(idata, kdata, odata), odata);
                }
        }
}

