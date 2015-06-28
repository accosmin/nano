#pragma once

#include "conv2d_make_toeplitz.hpp"
#include "nanocv/arch.h"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (using an already computed Toeplitz-like matrix)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv2d_toeplitz_buffered(const tmatrixi& idata,
                        const tmatrixk& kdata, const tmatrixo& toeplitz, tmatrixo& odata)
                {
                        NANOCV_UNUSED1_RELEASE(idata);

                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto osize = odata.size();
                        const auto ksize = kdata.size();

                        tensor::map_vector(odata.data(), osize) += toeplitz * tensor::map_vector(kdata.data(), ksize);
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

                        return conv2d_toeplitz_buffered(idata, kdata, make_toeplitz(idata, kdata, odata), odata);
                }
        }
}

