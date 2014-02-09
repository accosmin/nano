#ifndef NANOCV_OPENCL_SHARED_H
#define NANOCV_OPENCL_SHARED_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <memory>

namespace ncv
{
        namespace opencl
        {
                typedef std::shared_ptr<_cl_context>            rcontext_t;
                typedef std::shared_ptr<_cl_command_queue>      rqueue_t;
                typedef std::shared_ptr<_cl_program>            rprogram_t;
                typedef std::shared_ptr<_cl_kernel>             rkernel_t;
                typedef std::shared_ptr<_cl_mem>                rmem_t;

                ///
                /// \brief safely map to shared pointer a context
                ///
                rcontext_t make_shared(cl_context context);

                ///
                /// \brief safely map to shared pointer a command queue
                ///
                rqueue_t make_shared(cl_command_queue queue);

                ///
                /// \brief safely map to shared pointer a program
                ///
                rprogram_t make_shared(cl_program program);

                ///
                /// \brief safely map to shared pointer a kernel
                ///
                rkernel_t make_shared(cl_kernel kernel);

                ///
                /// \brief safely map to shared pointer a memory location
                ///
                rmem_t make_shared(cl_mem mem);
        }
}

#endif // NANOCV_OPENCL_SHARED_H
