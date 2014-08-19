#ifndef NANOCV_CUDA_H
#define NANOCV_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace ncv
{
        namespace cuda
        {
                ///
                /// \brief CUDA instance: stores information about CUDA devices.
                ///
                class manager_t
                {
                public:
                        
                        ///
                        /// \brief access to the only instance
                        ///
                        static const manager_t& instance();
                        
                        ///
                        /// \brief print CUDA system information
                        ///
                        bool print_info() const;
                        
                        ///
                        /// \brief count CUDA devices
                        ///
                        int count_devices() const { return m_devices; }
                        
                        ///
                        /// \brief get CUDA properties for a given device
                        ///
                        const cudaDeviceProp& get_device_properties(int device = 0) const
                        {
                                return m_properties[device]; 
                        }
                        
                private:
                        
                        ///
                        /// \brief constructor
                        ///
                        manager_t();
                        
                        ///
                        /// \brief disable copying
                        ///
                        manager_t(const manager_t&);
                        manager_t& operator=(const manager_t&);
                        
                private:
                        
                        // attributes
                        int                     m_devices;
                        cudaDeviceProp          m_properties[16];
                };
                
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
        }
}

#endif // NANOCV_CUDA_H

