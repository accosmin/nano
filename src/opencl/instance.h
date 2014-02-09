#ifndef NANOCV_OPENCL_INSTANCE_H
#define NANOCV_OPENCL_INSTANCE_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <memory>
#include <string>

namespace ncv
{
        namespace opencl
        {
                typedef std::shared_ptr<_cl_context>            rcontext_t;
                typedef std::shared_ptr<_cl_command_queue>      rqueue_t;
                typedef std::shared_ptr<_cl_program>            rprogram_t;
                typedef std::shared_ptr<_cl_kernel>             rkernel_t;

                ///
                /// \brief create (safe) shared context
                ///
                rcontext_t make_shared(cl_context context);

                ///
                /// \brief create (safe) shared command queue
                ///
                rqueue_t make_shared(cl_command_queue queue);

                ///
                /// \brief create (safe) shared program
                ///
                rprogram_t make_shared(cl_program program);

                ///
                /// \brief create (safe) shared kernel
                ///
                rkernel_t make_shared(cl_kernel kernel);

                ///
                /// \brief construct an OpenCL context
                ///
                rcontext_t make_context();

                ///
                /// \brief construct an OpenCL command queue on the given context
                ///
                rqueue_t make_command_queue(const rcontext_t& context, cl_device_id& device);

                ///
                /// \brief construct an OpenCL program on the given context from text
                ///
                rprogram_t make_program_from_text(const rcontext_t& context, cl_device_id device, const std::string& text);

                ///
                /// \brief construct an OpenCL program on the given context from source file
                ///
                rprogram_t make_program_from_file(const rcontext_t& context, cl_device_id device, const std::string& filename);

                ///
                /// \brief construct an OpenCL kernel from the given program with the given name
                ///
                rkernel_t make_kernel(const rprogram_t& program, const std::string& name);
        }
}

#endif // NANOCV_OPENCL_INSTANCE_H
