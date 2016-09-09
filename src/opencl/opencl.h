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
        namespace ocl
        {
                class manager_t;

                ///
                /// \brief map the given OpenCL error code to a string
                ///
                NANO_PUBLIC const char* error_string(const cl_int error);

                ///
                /// \brief map the given device type to a string
                ///
                NANO_PUBLIC const char* device_type_string(const cl_device_type type);

                ///
                /// \brief load text file (e.g. program/kernel source)
                ///
                NANO_PUBLIC std::string load_text_file(const std::string& filepath);

                ///
                /// \brief byte size
                ///
                template <typename ttensor>
                size_t bytesize(const ttensor& tensor)
                {
                        return tensor.size() * sizeof(typename ttensor::Scalar);
                }

                ///
                /// \brief OpenCL instance: platform information, manages devices, command queue, kernels and buffers.
                ///
                class NANO_PUBLIC manager_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        manager_t();

                        ///
                        /// \brief select the current device using the given type as a hint
                        ///
                        bool select(const cl_device_type type = CL_DEVICE_TYPE_GPU);

                        ///
                        /// \brief create OpenCL objects
                        ///
                        cl::Context make_context() const;
                        cl::CommandQueue make_command_queue(const cl::Context& context) const;
                        cl::Program make_program_from_file(const cl::Context& context, const std::string& filepath) const;
                        cl::Program make_program_from_text(const cl::Context& context, const std::string& source) const;
                        cl::Kernel make_kernel(const cl::Program& program, const std::string& name) const;
                        cl::Buffer make_buffer(const cl::Context& context, const size_t bytesize, const cl_mem_flags) const;

                private:

                        void select(const cl::Device& device);

                private:

                        // attributes
                        std::vector<cl::Platform>       m_platforms;    ///< available platforms
                        std::vector<cl::Device>         m_devices;      ///< available devices for all platforms
                        cl::Device                      m_device;       ///< selected device
                };
        }
}

