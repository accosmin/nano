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
                typedef std::shared_ptr<_cl_command_queue>      rqueue_t;
                typedef std::shared_ptr<_cl_context>            rcontext_t;
                typedef std::shared_ptr<_cl_program>            rprogram_t;
                typedef std::shared_ptr<_cl_kernel>             rkernel_t;

                rqueue_t make_shared(cl_command_queue queue);

                rcontext_t make_shared(cl_context context);

                rprogram_t make_shared(cl_program program);

                rkernel_t make_shared(cl_kernel kernel);

                rcontext_t make_context();

                rqueue_t make_command_queue(const rcontext_t& context, cl_device_id& device);

                rprogram_t make_program_from_text(const rcontext_t& context, cl_device_id device, const std::string& text);

                rprogram_t make_program_from_file(const rcontext_t& context, cl_device_id device, const std::string& filename);

                rkernel_t make_kernel(const rprogram_t& program, const std::string& kname);
        }
}

#endif // NANOCV_OPENCL_INSTANCE_H
