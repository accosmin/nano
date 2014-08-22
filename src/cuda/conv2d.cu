#include "conv2d.h"
#include "cuda.h"
#include <cstdio>

template
<
        typename tscalar
>
__global__ void kernel_conv2d(
        const tscalar* idata,
        const tscalar* kdata, int krows, int kcols,
        tscalar* odata, int orows, int ocols)
{
        const int c = threadIdx.x + blockIdx.x * blockDim.x;
        const int r = threadIdx.y + blockIdx.y * blockDim.y;
        
        if (r < orows && c < ocols)
        {
                const int icols = ocols + kcols - 1;

                tscalar sum = 0;
                for (int kr = 0; kr < krows; kr ++)
                {
                        for (int kc = 0; kc < kcols; kc ++)
                        {
                                sum += idata[(r + kr) * icols + (c + kc)] * kdata[kr * kcols + kc];
                        }
                }

                odata[r * ocols + c] = sum;
        }
}

template
<
        typename tscalar
>
__global__ void kernel_iconv2d(
        const tscalar* odata,
        const tscalar* kdata, int krows, int kcols,
        tscalar* idata, int irows, int icols)
{
        const int c = threadIdx.x + blockIdx.x * blockDim.x;
        const int r = threadIdx.y + blockIdx.y * blockDim.y;
        
        if (r < irows && c < icols)
        {
                const int orows = irows - krows + 1;
                const int ocols = icols - kcols + 1;

                const int krmin = max(0,     r - orows + 1);
                const int krmax = min(krows, r + 1);

                const int kcmin = max(0,     c - ocols + 1);
                const int kcmax = min(kcols, c + 1);

                tscalar sum = 0;
                for (int kr = krmin; kr < krmax; kr ++)
                {
                        for (int kc = kcmin; kc < kcmax; kc ++)
                        {
                                sum += odata[(r - kr) * ocols + (c - kc)] * kdata[kr * kcols + kc];
                        }
                }

                idata[r * icols + c] = sum;
        }
}

namespace ncv
{
        template
        <
                typename tscalar
        >
        static bool cuda_conv2d(
                const cuda::matrix_t<tscalar>& idata,
                const cuda::matrix_t<tscalar>& kdata,
                cuda::matrix_t<tscalar>& odata,
                int device, const cuda::stream_t* stream)
        {
                if (    odata.rows() + kdata.rows() != idata.rows() + 1 ||
                        odata.cols() + kdata.cols() != idata.cols() + 1)
                {
                        return false;
                }

                else
                {
                        const dim3 bsize = cuda::make_blocks2d(odata.rows(), odata.cols(), device);
                        const dim3 tsize = cuda::make_threads2d(odata.rows(), odata.cols(), device);

                        if (stream && stream->valid())
                        {
                                kernel_conv2d<<<bsize, tsize, 0, stream->data()>>>(
                                        idata.data(),
                                        kdata.data(), kdata.rows(), kdata.cols(),
                                        odata.data(), odata.rows(), odata.cols());
                        }
                        else
                        {
                                kernel_conv2d<<<bsize, tsize>>>(
                                        idata.data(),
                                        kdata.data(), kdata.rows(), kdata.cols(),
                                        odata.data(), odata.rows(), odata.cols());
                        }
 
                        return cudaGetLastError() == cudaSuccess;
                }
        }
        
        bool cuda::conv2d(const imatrix_t& idata, const imatrix_t& kdata, imatrix_t& odata, int device, const stream_t* stream)
        {
                return cuda_conv2d(idata, kdata, odata, device, stream);
        }

        bool cuda::conv2d(const fmatrix_t& idata, const fmatrix_t& kdata, fmatrix_t& odata, int device, const stream_t* stream)
        {
                return cuda_conv2d(idata, kdata, odata, device, stream);
        }

        bool cuda::conv2d(const dmatrix_t& idata, const dmatrix_t& kdata, dmatrix_t& odata, int device, const stream_t* stream)
        {
                return cuda_conv2d(idata, kdata, odata, device, stream);
        }

        template
        <
                typename tscalar
        >
        static bool cuda_iconv2d(
                const cuda::matrix_t<tscalar>& odata,
                const cuda::matrix_t<tscalar>& kdata,
                cuda::matrix_t<tscalar>& idata,
                int device, const cuda::stream_t* stream)
        {
                if (    odata.rows() + kdata.rows() != idata.rows() + 1 ||
                        odata.cols() + kdata.cols() != idata.cols() + 1)
                {
                        return false;
                }

                else
                {
                        const dim3 bsize = cuda::make_blocks2d(idata.rows(), idata.cols(), device);
                        const dim3 tsize = cuda::make_threads2d(idata.rows(), idata.cols(), device);

                        if (stream && stream->valid())
                        {
                                kernel_iconv2d<<<bsize, tsize, 0, stream->data()>>>(
                                        odata.data(),
                                        kdata.data(), kdata.rows(), kdata.cols(),
                                        idata.data(), idata.rows(), idata.cols());
                        }
                        else
                        {
                                kernel_iconv2d<<<bsize, tsize>>>(
                                        odata.data(),
                                        kdata.data(), kdata.rows(), kdata.cols(),
                                        idata.data(), idata.rows(), idata.cols());
                        }

                        return cudaGetLastError() == cudaSuccess;
                }
        }
        
        bool cuda::iconv2d(const imatrix_t& odata, const imatrix_t& kdata, imatrix_t& idata, int device, const stream_t* stream)
        {
                return cuda_iconv2d(odata, kdata, idata, device, stream);
        }

        bool cuda::iconv2d(const fmatrix_t& odata, const fmatrix_t& kdata, fmatrix_t& idata, int device, const stream_t* stream)
        {
                return cuda_iconv2d(odata, kdata, idata, device, stream);
        }

        bool cuda::iconv2d(const dmatrix_t& odata, const dmatrix_t& kdata, dmatrix_t& idata, int device, const stream_t* stream)
        {
                return cuda_iconv2d(odata, kdata, idata, device, stream);
        }
}
