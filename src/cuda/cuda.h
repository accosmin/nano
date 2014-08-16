#ifndef NANOCV_CUDA_H
#define NANOCV_CUDA_H

namespace ncv
{
        namespace cuda
        {
                ///
                /// \brief copy to and from device
                ///
                bool copyToDevice(const double* h_data, int size);
                bool copyFromDevice(const double* d_data, int size, double* h_data);
        }
}

#endif // NANOCV_CUDA_H

