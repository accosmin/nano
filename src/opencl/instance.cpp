#include "instance.h"
#include "util/logger.h"
#include <stdexcept>
#include <cassert>
#include <fstream>
#include <sstream>

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

        opencl::rcontext_t opencl::make_context()
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

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rqueue_t opencl::make_command_queue(const rcontext_t& context, cl_device_id& device)
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

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rprogram_t opencl::make_program_from_text(const rcontext_t& context, cl_device_id device, const std::string& text)
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

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rprogram_t opencl::make_program_from_file(const rcontext_t& context, cl_device_id device, const std::string& filename)
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

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rkernel_t opencl::make_kernel(const rprogram_t& program, const std::string& kname)
        {
                rkernel_t kernel = make_shared(clCreateKernel(program.get(), kname.c_str(), NULL));
                if (!kernel)
                {
                        throw std::runtime_error("Failed to create kernel <" + kname + ">!");
                }

                return kernel;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
