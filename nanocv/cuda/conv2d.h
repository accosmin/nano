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
                template
                <       
                        typename tscalar
                >
                bool conv2d(const matrix_t<tscalar>& idata, const matrix_t<tscalar>& kdata, 
                            matrix_t<tscalar>& odata, int device = 0);
                
                template <>
                bool conv2d<int>(const imatrix_t&, const imatrix_t&, imatrix_t&, int);
                template <>
                bool conv2d<float>(const fmatrix_t&, const fmatrix_t&, fmatrix_t&, int);
                template <>
                bool conv2d<double>(const dmatrix_t&, const dmatrix_t&, dmatrix_t&, int);

                ///
                /// \brief inverse 2D convolution: idata = odata @ kdata
                ///
                template
                <       
                        typename tscalar
                >                
                bool iconv2d(const matrix_t<tscalar>& odata, const matrix_t<tscalar>& kdata, 
                             matrix_t<tscalar>& idata, int device = 0);
                
                template <>
                bool iconv2d<int>(const imatrix_t&, const imatrix_t&, imatrix_t&, int);
                template <>
                bool iconv2d<float>(const fmatrix_t&, const fmatrix_t&, fmatrix_t&, int);
                template <>
                bool iconv2d<double>(const dmatrix_t&, const dmatrix_t&, dmatrix_t&, int);
        }
}

#endif // NANOCV_CUDA_CONV2D_H

