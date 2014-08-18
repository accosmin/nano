#ifndef NANOCV_CUDA_CONV2D_H
#define NANOCV_CUDA_CONV2D_H

#include "matrix.hpp"

namespace ncv
{
        namespace cuda
        {
                ///
                /// \brief 2D convolution: odata = idata @ kdata
                ///
                bool conv2f(const fmatrix_t& idata, const fmatrix_t& kdata, fmatrix_t& odata, int device = 0);
                bool conv2d(const dmatrix_t& idata, const dmatrix_t& kdata, dmatrix_t& odata, int device = 0);

                ///
                /// \brief inverse 2D convolution: idata = odata @ kdata
                ///
                bool iconv2f(const fmatrix_t& odata, const fmatrix_t& kdata, fmatrix_t& idata, int device = 0);
                bool iconv2d(const dmatrix_t& odata, const dmatrix_t& kdata, dmatrix_t& idata, int device = 0);
        }
}

#endif // NANOCV_CUDA_CONV2D_H

