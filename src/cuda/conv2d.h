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
                bool conv2d(const matrix_t<double>& idata, const matrix_t<double>& kdata, matrix_t<double>& odata,
                            int device = 0);

                ///
                /// \brief inverse 2D convolution: idata = odata @ kdata
                ///
                bool iconv2d(const matrix_t<double>& odata, const matrix_t<double>& kdata, matrix_t<double>& idata,
                             int device = 0);
        }
}

#endif // NANOCV_CUDA_CONV2D_H

