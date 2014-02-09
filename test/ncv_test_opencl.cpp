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

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b)
{
        memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * ARRAY_SIZE, a, NULL);
        memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * ARRAY_SIZE, b, NULL);
        memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * ARRAY_SIZE, NULL, NULL);

        if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
        {
                std::cerr << "Error creating memory objects." << std::endl;
                return false;
        }

        return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_mem memObjects[3])
{
        for (int i = 0; i < 3; i++)
        {
                if (memObjects[i] != 0)
                        clReleaseMemObject(memObjects[i]);
        }
}

int main(int argc, char *argv[])
{
        using namespace ncv;

        cl_device_id device = 0;

        opencl::rcontext_t context = opencl::make_context();
        opencl::rqueue_t commandQueue = opencl::make_command_queue(context, device);
        opencl::rprogram_t program = opencl::make_program_from_text(context, device, program_source);
        opencl::rkernel_t kernel = opencl::make_kernel(program, "hello_kernel");

        cl_mem memObjects[3] = { 0, 0, 0 };
        cl_int errNum;

        // Create memory objects that will be used as arguments to
        // kernel.  First create host memory arrays that will be
        // used to store the arguments to the kernel
        float result[ARRAY_SIZE];
        float a[ARRAY_SIZE];
        float b[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++)
        {
                a[i] = (float)i;
                b[i] = (float)(i * 2);
        }

        if (!CreateMemObjects(context.get(), memObjects, a, b))
        {
                Cleanup(memObjects);
                return 1;
        }

        // Set the kernel arguments (result, a, b)
        errNum = clSetKernelArg(kernel.get(), 0, sizeof(cl_mem), &memObjects[0]);
        errNum |= clSetKernelArg(kernel.get(), 1, sizeof(cl_mem), &memObjects[1]);
        errNum |= clSetKernelArg(kernel.get(), 2, sizeof(cl_mem), &memObjects[2]);
        if (errNum != CL_SUCCESS)
        {
                std::cerr << "Error setting kernel arguments." << std::endl;
                Cleanup(memObjects);
                return 1;
        }

        size_t globalWorkSize[1] = { ARRAY_SIZE };
        size_t localWorkSize[1] = { 1 };

        // Queue the kernel up for execution across the array
        errNum = clEnqueueNDRangeKernel(commandQueue.get(), kernel.get(), 1, NULL,
                                        globalWorkSize, localWorkSize,
                                        0, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
                std::cerr << "Error queuing kernel for execution." << std::endl;
                Cleanup(memObjects);
                return 1;
        }

        // Read the output buffer back to the Host
        errNum = clEnqueueReadBuffer(commandQueue.get(), memObjects[2], CL_TRUE,
                        0, ARRAY_SIZE * sizeof(float), result,
                        0, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
                std::cerr << "Error reading result buffer." << std::endl;
                Cleanup(memObjects);
                return 1;
        }

        // Output the result buffer
        for (int i = 0; i < ARRAY_SIZE; i++)
        {
                std::cout << result[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Executed program succesfully." << std::endl;
        Cleanup(memObjects);

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
