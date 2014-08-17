#ifndef NANOCV_CUDA_H
#define NANOCV_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>

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
                /// \brief allocated (array of doubles) buffer on the device.
                ///
                /// NB: it would be nice to use thrust::device_vector<> directly!
                ///
                struct device_buffer_impl_t;
                class device_buffer_t
                {
                public:

                        ///
                        /// \brief constructor (allocate the given number of doubles on the device)
                        ///
                        device_buffer_t(int size);

                        ///
                        /// \brief disable copying
                        ///
                        device_buffer_t(const device_buffer_t&);
                        device_buffer_t& operator=(const device_buffer_t&);

                        ///
                        /// \brief destructor
                        ///
                        ~device_buffer_t();

                        ///
                        /// \brief to device
                        ///
                        bool copyToDevice(const double* h_data) const;

                        ///
                        /// \brief from device
                        ///
                        bool copyFromDevice(double* h_data) const;

                        ///
                        /// \brief access functions
                        ///
                        int size() const;
                        bool empty() const;

                        const device_buffer_impl_t& get() const;
                        device_buffer_impl_t& get();

                private:

                        // attributes
                        device_buffer_impl_t*   m_impl;
                };

                bool addbsquared(const device_buffer_t& a, const device_buffer_t& b, device_buffer_t& c);
        }
}

#endif // NANOCV_CUDA_H

