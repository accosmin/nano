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

namespace nano
{
namespace ocl
{
        ///
        /// \brief select the current device using the given type as a hint.
        ///
        NANO_PUBLIC bool select(const cl_device_type type = CL_DEVICE_TYPE_GPU);

        ///
        /// \brief current command queue.
        ///
        NANO_PUBLIC cl::CommandQueue& queue();

        ///
        /// \brief tensor size in bytes.
        ///
        template <typename ttensor>
        std::size_t tensor_bytes(const ttensor& t)
        {
                return static_cast<std::size_t>(t.size()) * sizeof(typename ttensor::Scalar);
        }

        ///
        /// \brief create a buffer to hold the given tensor.
        ///
        template <typename ttensor>
        cl::Buffer make_buffer(const ttensor& t, const cl_mem_flags flags)
        {
                return make_buffer(tensor_bytes(t), flags);
        }

        ///
        /// \brief create a buffer to hold the given number of bytes.
        ///
        template <>
        NANO_PUBLIC cl::Buffer make_buffer<std::size_t>(const std::size_t& bytesize, const cl_mem_flags);

        ///
        /// \brief read a device buffer to the given tensor (blocking operation).
        ///
        template <typename ttensor>
        cl_int read(const cl::Buffer& buffer, ttensor& t)
        {
                return queue().enqueueReadBuffer(buffer, CL_TRUE, 0, tensor_bytes(t), t.data());
        }

        ///
        /// \brief write to a device buffer from the given tensor (blocking operation).
        ///
        template <typename ttensor>
        cl_int write(const cl::Buffer& buffer, const ttensor& t)
        {
                return queue().enqueueWriteBuffer(buffer, CL_TRUE, 0, tensor_bytes(t), t.data());
        }

        ///
        /// \brief create & compile a program from source file.
        ///
        NANO_PUBLIC cl::Program make_program_from_file(const std::string& filepath);

        ///
        /// \brief create & compile a program from memory.
        ///
        NANO_PUBLIC cl::Program make_program_from_text(const std::string& source);

        ///
        /// \brief create a kernel from a program.
        ///
        NANO_PUBLIC cl::Kernel make_kernel(const cl::Program& program, const char* name);

        ///
        /// \brief retrieve a built-in kernel.
        ///
        NANO_PUBLIC cl::Kernel make_kernel(const char* name);

        ///
        /// \brief map the given OpenCL error code to a string.
        ///
        NANO_PUBLIC const char* error_string(const cl_int error);

        ///
        /// \brief map the given device type to a string.
        ///
        NANO_PUBLIC const char* device_type_string(const cl_device_type type);

        namespace detail
        {
                template <typename targ>
                void set_args(cl::Kernel& kernel, const cl_uint index, const targ& arg)
                {
                        kernel.setArg(index, arg);
                }

                template <typename targ, typename... targs>
                void set_args(cl::Kernel& kernel, const cl_uint index, const targ& arg, const targs&... args)
                {
                        kernel.setArg(index, arg);
                        set_args(kernel, index + 1, args...);
                }
        }

        ///
        /// \brief set arguments to a kernel.
        ///
        template <typename... targs>
        void set_args(cl::Kernel& kernel, targs... args)
        {
                detail::set_args(kernel, 0, args...);
        }
}
}

