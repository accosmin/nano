#pragma once

#undef CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>        // NB: may need to manually install/copy the C++ wrapper from kronos' website!
#else
#include <CL/cl2.hpp>
#endif

#include "arch.h"
#include <vector>
#include <string>

namespace nano
{
        ///
        /// \brief map the given OpenCL error code to a string
        ///
        NANO_PUBLIC const char* error_string(const cl_int error);

        ///
        /// \brief map the given device type to a string
        ///
        NANO_PUBLIC const char* device_type_string(const cl_device_type type);

        ///
        /// \brief OpenCL instance: platform information, manages devices, command queue, kernels and buffers.
        ///
        class NANO_PUBLIC opencl_manager_t
        {
        public:

                ///
                /// \brief initialize OpenCL platforms & devices.
                ///
                void init();

                ///
                /// \brief select the current device using the given type as a hint.
                ///
                bool select(const cl_device_type type = CL_DEVICE_TYPE_GPU);

                ///
                /// \brief create a buffer to hold the given tensor.
                ///
                template <typename ttensor>
                cl::Buffer make_buffer(const ttensor& t, const cl_mem_flags flags) const
                {
                        return make_buffer(tensor_size(t), flags);
                }

                ///
                /// \brief read a device buffer to the given tensor (blocking operation).
                ///
                template <typename ttensor>
                cl_int read(const cl::Buffer& buffer, ttensor& t) const
                {
                        return m_command_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, tensor_size(t), t.data());
                }

                ///
                /// \brief write to a device buffer from the given tensor (blocking operation).
                ///
                template <typename ttensor>
                cl_int write(const cl::Buffer& buffer, const ttensor& t) const
                {
                        return m_command_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, tensor_size(t), t.data());
                }

                ///
                /// \brief create OpenCL objects.
                ///
                cl::Program make_program_from_file(const std::string& filepath) const;
                cl::Program make_program_from_text(const std::string& source) const;
                cl::Kernel make_kernel(const cl::Program& program, const std::string& name) const;
                cl::Buffer make_buffer(const size_t bytesize, const cl_mem_flags) const;

                ///
                /// \brief current context.
                ///
                cl::Context& context() { return m_context; }

                ///
                /// \brief current command queue.
                ///
                cl::CommandQueue& command_queue() { return m_command_queue; }

        private:

                void select(const cl::Device& device);

                template <typename ttensor>
                static size_t tensor_size(const ttensor& t)
                {
                        return static_cast<size_t>(t.size()) * sizeof(typename ttensor::Scalar);
                }

        private:

                // attributes
                std::vector<cl::Platform>       m_platforms;    ///< available platforms
                std::vector<cl::Device>         m_devices;      ///< available devices for all platforms
                cl::Device                      m_device;       ///< selected device
                cl::Context                     m_context;      ///< context for the selected device
                cl::CommandQueue                m_command_queue;///< command queue for the selected contex
        };
}

