#include "ncv.h"
#include "opencl/instance.h"

#include <iostream>
#include <fstream>
#include <sstream>

///
//  Constants
//
const int ARRAY_SIZE = 1000;

const char* program_source = "\n" \
"__kernel void hello_kernel(                            \n" \
"       __global const float* a,                        \n" \
"       __global const float* b,                        \n" \
"       __global float* result)                         \n" \
"{                                                      \n" \
"       int gid = get_global_id(0);                     \n" \
"       result[gid] = a[gid] + b[gid];                  \n" \
"}                                                      \n" \
"\n";

int main(int argc, char *argv[])
{
        using namespace ncv;

        cl_device_id device = 0;

        opencl::rcontext_t context = opencl::make_context();
        opencl::rqueue_t commandQueue = opencl::make_command_queue(context, device);
        opencl::rprogram_t program = opencl::make_program_from_text(context, device, program_source);
        opencl::rkernel_t kernel = opencl::make_kernel(program, "hello_kernel");

        // Create memory objects that will be used as arguments to kernel.
        // First create host memory arrays that will be used to store the arguments to the kernel
        float c[ARRAY_SIZE];
        float a[ARRAY_SIZE];
        float b[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++)
        {
                a[i] = (float)i;
                b[i] = (float)(i * 2);
        }

        opencl::rmem_t mema = opencl::make_read_mem(context, a, ARRAY_SIZE);
        opencl::rmem_t memb = opencl::make_read_mem(context, b, ARRAY_SIZE);
        opencl::rmem_t memc = opencl::make_write_mem(context, c, ARRAY_SIZE);

        // Set the kernel arguments (c, a, b)
        if (    !opencl::set_argument(kernel, 0, mema) ||
                !opencl::set_argument(kernel, 1, memb) ||
                !opencl::set_argument(kernel, 2, memc))
        {
                std::cerr << "Error setting kernel arguments." << std::endl;
                return 1;
        }

        size_t globalWorkSize[1] = { ARRAY_SIZE };
        size_t localWorkSize[1] = { 1 };

        // Queue the kernel up for execution across the array
        if (    clEnqueueNDRangeKernel(
                commandQueue.get(), kernel.get(), 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) != CL_SUCCESS)
        {
                std::cerr << "Error queuing kernel for execution." << std::endl;
                return 1;
        }

        // Read the output buffer back to the Host
        if (    clEnqueueReadBuffer(
                commandQueue.get(), memc.get(), CL_TRUE, 0, ARRAY_SIZE * sizeof(float), c, 0, NULL, NULL) != CL_SUCCESS)
        {
                std::cerr << "Error reading result buffer." << std::endl;
                return 1;
        }

        // Output the result buffer
        for (int i = 0; i < ARRAY_SIZE; i++)
        {
                std::cout << c[i] << " ";
        }
        std::cout << std::endl;

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
