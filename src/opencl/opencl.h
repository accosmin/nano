#pragma once

#undef CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

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
                NANO_PUBLIC const char* error_string(cl_int error);

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
                        /// \brief check if an OpenCL instance was correctly loaded
                        ///
                        bool valid() const { return !m_platforms.empty() && !m_devices.empty(); }

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

                        // attributes
                        std::vector<cl::Platform>       m_platforms;
                        std::vector<cl::Device>         m_devices;
                };
        }
}

