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
        /// \brief set arguments to an OpenCL kernel.
        ///
        template <typename... targs>
        void set_args(cl::Kernel& kernel, targs... args)
        {
                detail::set_args(kernel, 0, args...);
        }
}

