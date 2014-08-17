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
                cudaDeviceProp get_device_properties(int device = 0);

                ///
                /// \brief c[i] = a[i] + b[i] * b[i] (test kernel)
                ///
                bool addbsquared(const vector_t<double>& a, const vector_t<double>& b, vector_t<double>& c);
        }
}

#endif // NANOCV_CUDA_H

