#ifndef NANOCV_CUDA_H
#define NANOCV_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.hpp"

namespace ncv
{
        namespace cuda
        {
                ///
                /// \brief print CUDA system information
                ///
                bool print_info();

                ///
                /// \brief count CUDA devices
                ///
                int count_devices();

                ///
                /// \brief get CUDA properties for a given device
                ///
                cudaDeviceProp get_device_properties(int device);

                ///
                /// \brief construct the optimal number of blocks for 1D & 2D processing
                ///
                dim3 make_block1d_count(int size, int device = 0);
                dim3 make_block2d_count(int rows, int cols, int device = 0);

                ///
                /// \brief construct the optimal block size (= number of threads per block) for 1D & 2D processing
                ///
                dim3 make_block1d_size(int size, int device = 0);
                dim3 make_block2d_size(int rows, int cols, int device = 0);

                ///
                /// \brief c[i] = a[i] + b[i] * b[i] (test kernel)
                ///
                bool addbsquared(const vector_t<double>& a, const vector_t<double>& b, vector_t<double>& c, int device = 0);
        }
}

#endif // NANOCV_CUDA_H

