#include "ncv.h"

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace ncv
{
        namespace opencl
        {
                typedef std::shared_ptr<_cl_command_queue>      rqueue_t;
                typedef std::shared_ptr<_cl_context>            rcontext_t;
                typedef std::shared_ptr<_cl_program>            rprogram_t;
                typedef std::shared_ptr<_cl_kernel>             rkernel_t;

                rqueue_t make_shared(cl_command_queue queue)
                {
                        return rqueue_t(queue, [] (cl_command_queue queue)
                        {
                                if (queue)
                                {
                                        std::cout << "ncv::opencl:: release command queue" << std::endl;
                                        clReleaseCommandQueue(queue);
                                }
                        });
                }

                rcontext_t make_shared(cl_context context)
                {
                        return rcontext_t(context, [] (cl_context context)
                        {
                                if (context)
                                {
                                        std::cout << "ncv::opencl:: release context" << std::endl;
                                        clReleaseContext(context);
                                }
                        });
                }

                rprogram_t make_shared(cl_program program)
                {
                        return rprogram_t(program, [] (cl_program program)
                        {
                                if (program)
                                {
                                        std::cout << "ncv::opencl:: release program" << std::endl;
                                        clReleaseProgram(program);
                                }
                        });
                }

                rkernel_t make_shared(cl_kernel kernel)
                {
                        return rkernel_t(kernel, [] (cl_kernel kernel)
                        {
                                if (kernel)
                                {
                                        std::cout << "ncv::opencl:: release kernel" << std::endl;
                                        clReleaseKernel(kernel);
                                }
                        });
                }

                rcontext_t make_context()
                {
                        // First, select an OpenCL platform to run on.  For this example, we
                        // simply choose the first available platform.  Normally, you would
                        // query for all available platforms and select the most appropriate one.
                        cl_uint numPlatforms;
                        cl_platform_id firstPlatformId;
                        cl_int errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
                        if (errNum != CL_SUCCESS || numPlatforms <= 0)
                        {
                                throw std::runtime_error("Failed to find any OpenCL platforms!");
                        }

                        // Next, create an OpenCL context on the platform.  Attempt to
                        // create a GPU-based context, and if that fails, try to create
                        // a CPU-based context.
                        cl_context_properties contextProperties[] =
                        {
                                CL_CONTEXT_PLATFORM,
                                (cl_context_properties)firstPlatformId,
                                0
                        };

                        rcontext_t context = make_shared(clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum));
                        if (errNum != CL_SUCCESS)
                        {
                                log_warning() << "Could not create GPU context, trying CPU...";
                                context = make_shared(clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum));
                                if (errNum != CL_SUCCESS)
                                {
                                        throw std::runtime_error("Failed to create an OpenCL GPU or CPU context!");
                                }
                        }

                        return context;
                }

                rqueue_t make_command_queue(const rcontext_t& context, cl_device_id& device)
                {
                        cl_int errNum;
                        cl_device_id *devices;
                        size_t deviceBufferSize = -1;

                        // First get the size of the devices buffer
                        errNum = clGetContextInfo(context.get(), CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
                        if (errNum != CL_SUCCESS)
                        {
                                throw std::runtime_error("Failed to query OpenCL context information!");
                        }

                        if (deviceBufferSize <= 0)
                        {
                                throw std::runtime_error("No OpenCL devices available!");
                        }

                        // Allocate memory for the devices buffer
                        devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
                        errNum = clGetContextInfo(context.get(), CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
                        if (errNum != CL_SUCCESS)
                        {
                                delete [] devices;
                                throw std::runtime_error("Failed to get OpenCL device IDs!");
                        }

                        // In this example, we just choose the first available device.  In a
                        // real program, you would likely use all available devices or choose
                        // the highest performance device based on OpenCL device queries
                        rqueue_t queue = make_shared(clCreateCommandQueue(context.get(), devices[0], 0, NULL));
                        if (!queue)
                        {
                                delete [] devices;
                                throw std::runtime_error("Failed to create OpenCL command queue for device 0!");
                        }

                        device = devices[0];
                        delete [] devices;

                        return queue;
                }

                rprogram_t make_program_from_text(const rcontext_t& context, cl_device_id device, const string_t& text)
                {
                        const char* str = text.c_str();
                        rprogram_t program = make_shared(clCreateProgramWithSource(context.get(), 1, (const char**)&str, NULL, NULL));
                        if (!program)
                        {
                                throw std::runtime_error("Failed to create OpenCL program from source!");
                        }

                        cl_int errNum = clBuildProgram(program.get(), 0, NULL, NULL, NULL, NULL);
                        if (errNum != CL_SUCCESS)
                        {
                                // Determine the reason for the error
                                char buildLog[16384];
                                clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG,
                                                      sizeof(buildLog), buildLog, NULL);

                                log_error() << "Error in kernel: ";
                                log_error() << buildLog;
                                program.reset();
                        }

                        return program;
                }

                rprogram_t make_program_from_file(const rcontext_t& context, cl_device_id device, const string_t& filename)
                {
                        std::ifstream kernelFile(filename, std::ios::in);
                        if (!kernelFile.is_open())
                        {
                                throw std::runtime_error("Failed to open OpenCL program file <" + filename + ">!");
                        }

                        std::ostringstream oss;
                        oss << kernelFile.rdbuf();
                        return make_program_from_text(context, device, oss.str());
                }

                rkernel_t make_kernel(const rprogram_t& program, const string_t& kname)
                {
                        rkernel_t kernel = make_shared(clCreateKernel(program.get(), kname.c_str(), NULL));
                        if (!kernel)
                        {
                                throw std::runtime_error("Failed to create kernel <" + kname + ">!");
                        }

                        return kernel;
                }
        }
}

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
