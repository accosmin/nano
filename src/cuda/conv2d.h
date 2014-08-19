#ifndef NANOCV_CUDA_CONV2D_H
#define NANOCV_CUDA_CONV2D_H

#include "matrix.hpp"
#include <boost/concept_check.hpp>

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
                bool conv2d<int>(const imatrix_t& idata, const imatrix_t& kdata, imatrix_t& odata, int device);
                template <>
                bool conv2d<float>(const fmatrix_t& idata, const fmatrix_t& kdata, fmatrix_t& odata, int device);
                template <>
                bool conv2d<double>(const dmatrix_t& idata, const dmatrix_t& kdata, dmatrix_t& odata, int device);                

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
                bool iconv2d<int>(const imatrix_t& odata, const imatrix_t& kdata, imatrix_t& idata, int device);
                template <>
                bool iconv2d<float>(const fmatrix_t& odata, const fmatrix_t& kdata, fmatrix_t& idata, int device);
                template <>
                bool iconv2d<double>(const dmatrix_t& odata, const dmatrix_t& kdata, dmatrix_t& idata, int device);
        }
}

#endif // NANOCV_CUDA_CONV2D_H

