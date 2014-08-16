#ifndef NANOCV_CUDA_H
#define NANOCV_CUDA_H

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

namespace ncv
{
        namespace cuda
        {
                ///
                /// \brief copy to and from device
                ///
                bool copyToDevice(const double* h_data, int size, double* d_data);
                bool copyFromDevice(const double* d_data, int size, double* h_data);
        }
}

#endif // NANOCV_CUDA_H

