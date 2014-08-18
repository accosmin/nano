#include "conv2d.h"
#include "cuda.h"

__global__ void kernel_conv2d(
        const ncv::cuda::matrix_t<double>& idata,
        const ncv::cuda::matrix_t<double>& kdata,
        ncv::cuda::matrix_t<double>& odata)
{
        const int c = threadIdx.x + blockIdx.x * blockDim.x;
        const int r = threadIdx.y + blockIdx.y * blockDim.y;
        
        const int orows = odata.rows();
        const int ocols = odata.cols();

        if (r < orows && c < ocols)
        {
                const int krows = kdata.rows();
                const int kcols = kdata.cols();

                double sum = 0;
                for (int kr = 0; kr < krows; kr ++)
                {
                        for (int kc = 0; kc < kcols; kc ++)
                        {
                                sum += idata(r + kr, c + kc) * kdata(kr, kc);
                        }
                }

                odata(r, c) = sum;
        }
}

__global__ void kernel_iconv2d(
        const ncv::cuda::matrix_t<double>& odata,
        const ncv::cuda::matrix_t<double>& kdata,
        ncv::cuda::matrix_t<double>& idata)
{
        const int c = threadIdx.x + blockIdx.x * blockDim.x;
        const int r = threadIdx.y + blockIdx.y * blockDim.y;
        
        const int irows = idata.rows();
        const int icols = idata.cols();

        if (r < irows && c < icols)
        {
                const int krows = kdata.rows();
                const int kcols = kdata.cols();
                
                const int orows = irows - krows + 1;
                const int ocols = icols - kcols + 1;

                const int krmin = max(0,     r - orows + 1);
                const int krmax = min(krows, r + 1);

                const int kcmin = max(0,     c - ocols + 1);
                const int kcmax = min(kcols, c + 1);

                double sum = 0;
                for (int kr = krmin; kr < krmax; kr ++)
                {
                        for (int kc = kcmin; kc < kcmax; kc ++)
                        {
                                sum += odata(r - kr, c - kc) * kdata(kr, kc);
                        }
                }

                idata(r, c) = sum;
        }
}

namespace ncv
{
        bool cuda::conv2d(
                const matrix_t<double>& idata, const matrix_t<double>& kdata, matrix_t<double>& odata,
                int device)
        {
                if (    odata.rows() + kdata.rows() != idata.rows() + 1 ||
                        odata.cols() + kdata.cols() != idata.cols() + 1)
                {
                        return false;
                }

                else
                {
                        const dim3 ksize = cuda::make_block2d_count(odata.rows(), odata.cols(), device);
                        const dim3 bsize = cuda::make_block2d_size(odata.rows(), odata.cols(), device);

                        kernel_conv2d<<<ksize, bsize>>>(idata, kdata, odata);

                        return cudaGetLastError() == cudaSuccess;
                }
        }

        bool cuda::iconv2d(
                const matrix_t<double>& odata, const matrix_t<double>& kdata, matrix_t<double>& idata,
                int device)
        {
                if (    odata.rows() + kdata.rows() != idata.rows() + 1 ||
                        odata.cols() + kdata.cols() != idata.cols() + 1)
                {
                        return false;
                }

                else
                {
                        const dim3 ksize = cuda::make_block2d_count(idata.rows(), idata.cols(), device);
                        const dim3 bsize = cuda::make_block2d_size(idata.rows(), idata.cols(), device);

                        kernel_iconv2d<<<ksize, bsize>>>(odata, kdata, idata);

                        return cudaGetLastError() == cudaSuccess;
                }
        }
}
