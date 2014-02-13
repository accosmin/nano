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
                /// \brief OpenCL instance: manages devices, command queue, kernels and buffers
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
                        /// \brief access the context
                        ///
                        const cl::Context& context() const { return m_context; }

                        ///
                        /// \brief access the command queue
                        ///
                        const cl::CommandQueue& queue() const { return m_queue; }

                        ///
                        /// \brief build program from file
                        ///
                        cl::Program program_from_file(const std::string& filepath) const;

                        ///
                        /// \brief build program from source text
                        ///
                        cl::Program program_from_text(const std::string& source) const;

                private:

                        // attributes
                        std::vector<cl::Platform>       m_platforms;
                        std::vector<cl::Device>         m_devices;
                        cl::Context                     m_context;
                        cl::CommandQueue                m_queue;
                };
        }
}

#endif // NANOCV_OPENCL_H
