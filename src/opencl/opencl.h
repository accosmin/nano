#ifndef NANOCV_OPENCL_H
#define NANOCV_OPENCL_H

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif

#include "common/singleton.hpp"

namespace ncv
{
        namespace ocl
        {                
                ///
                /// \brief map the given OpenCL error code to a string
                ///
                const char* error_string(cl_int error);

                ///
                /// \brief load text file (e.g. program/kernel source)
                ///
                std::string load_text_file(const std::string& filepath);

                ///
                /// \brief byte size
                ///
                template
                <
                        typename ttensor
                >
                size_t bytesize(const ttensor& tensor)
                {
                        return tensor.size() * sizeof(typename ttensor::Scalar);
                }

                ///
                /// \brief OpenCL instance: platform information, manages devices, command queue, kernels and buffers.
                ///
                class manager_t : public singleton_t<manager_t>
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
                        cl::Buffer make_buffer(const cl::Context& context, size_t bytesize, int flags) const;

                private:

                        // attributes
                        std::vector<cl::Platform>       m_platforms;
                        std::vector<cl::Device>         m_devices;
                };
        }
}

#endif // NANOCV_OPENCL_H
