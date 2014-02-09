#include "cl_shared.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rqueue_t opencl::make_shared(cl_command_queue queue)
        {
                return rqueue_t(queue, [] (cl_command_queue queue)
                {
                        if (queue)
                        {
                                clReleaseCommandQueue(queue);
                        }
                });
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rcontext_t opencl::make_shared(cl_context context)
        {
                return rcontext_t(context, [] (cl_context context)
                {
                        if (context)
                        {
                                clReleaseContext(context);
                        }
                });
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rprogram_t opencl::make_shared(cl_program program)
        {
                return rprogram_t(program, [] (cl_program program)
                {
                        if (program)
                        {
                                clReleaseProgram(program);
                        }
                });
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rkernel_t opencl::make_shared(cl_kernel kernel)
        {
                return rkernel_t(kernel, [] (cl_kernel kernel)
                {
                        if (kernel)
                        {
                                clReleaseKernel(kernel);
                        }
                });
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rmem_t opencl::make_shared(cl_mem mem)
        {
                return rmem_t(mem, [] (cl_mem mem)
                {
                        if (mem)
                        {
                                clReleaseMemObject(mem);
                        }
                });
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
