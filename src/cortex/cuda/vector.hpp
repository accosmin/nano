#pragma once

#include "stream.h"

namespace nano
{
        namespace cuda
        {
                ///
                /// \brief allocated 1D buffer on the device.
                ///
                template
                <
                        typename tscalar
                >
                class vector_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        vector_t(int size = 0)
                                :       m_data(NULL),
                                        m_size(0)
                        {
                                const cudaError status = cudaMalloc((void**)&m_data, size * sizeof(tscalar));
                                if (status == cudaSuccess)
                                {
                                        m_size = size;
                                }
                        }

                        ///
                        /// \brief disable copying
                        ///
                        vector_t(const vector_t&);
                        vector_t& operator=(const vector_t&);

                        ///
                        /// \brief destructor
                        ///
                        virtual ~vector_t()
                        {
                                cudaFree(m_data);
                        }

                        ///
                        /// \brief resize to new dimensions
                        ///
                        bool resize(int size)
                        {
                                cudaFree(m_data);
                                m_data = NULL;

                                const cudaError status = cudaMalloc((void**)&m_data, size * sizeof(tscalar));
                                if (status == cudaSuccess)
                                {
                                        m_size = size;
                                        return true;
                                }
                                else
                                {
                                        return false;
                                }
                        }

                        ///
                        /// \brief to device
                        ///
                        cudaError to_device(const tscalar* h_data) const
                        {
                                return cudaMemcpy(m_data, h_data, size() * sizeof(tscalar),
                                                  cudaMemcpyHostToDevice);
                        }

                        ///
                        /// \brief to device (using a stream)
                        ///
                        cudaError to_device(const tscalar* h_data, const stream_t& stream) const
                        {
                                return cudaMemcpyAsync(m_data, h_data, size() * sizeof(tscalar),
                                                       cudaMemcpyHostToDevice, stream.data());
                        }

                        ///
                        /// \brief from device
                        ///
                        cudaError from_device(tscalar* h_data) const
                        {
                                return cudaMemcpy(h_data, m_data, size() * sizeof(tscalar),
                                                  cudaMemcpyDeviceToHost);
                        }

                        ///
                        /// \brief from device (using a stream)
                        ///
                        cudaError from_device(tscalar* h_data, const stream_t& stream) const
                        {
                                return cudaMemcpyAsync(h_data, m_data, size() * sizeof(tscalar),
                                                       cudaMemcpyDeviceToHost, stream.data());
                        }

                        ///
                        /// \brief access functions
                        ///
                        int size() const { return m_size; }
                        bool empty() const { return m_data != NULL && m_size > 0; }

                        tscalar* data() { return m_data; }
                        const tscalar* data() const { return m_data; }

                        tscalar operator()(int i) const { return m_data[i]; }
                        tscalar& operator()(int i) { return m_data[i]; }

                protected:

                        // attributes
                        tscalar*                m_data;
                        int                     m_size;
                };

                typedef vector_t<int>           ivector_t;
                typedef vector_t<float>         fvector_t;
                typedef vector_t<double>        dvector_t;
        }
}

