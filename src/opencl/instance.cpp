#include "instance.h"
#include "util/logger.h"
#include <cassert>
#include <fstream>
#include <sstream>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rcontext_t opencl::make_context()
        {
                cl_platform_id platform;
                if (clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS)
                {
                        log_error() << "Failed to find any OpenCL platform!";
                        return rcontext_t();
                }

                cl_device_id device;
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) != CL_SUCCESS)
                {
                        log_error() << "Failed to find any OpenCL GPU device!";
                        return rcontext_t();
                }

                {
                        char vendor[1024];
                        std::fill(vendor, vendor + 1024, '\0');
                        clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);

                        char name[1024];
                        std::fill(name, name + 1024, '\0');
                        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);

                        char driver[1024];
                        std::fill(driver, driver + 1024, '\0');
                        clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driver), driver, NULL);

                        char version[1024];
                        std::fill(version, version + 1024, '\0');
                        clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);

                        cl_ulong gmemsize, lmemsize;
                        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmemsize), &gmemsize, NULL);
                        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmemsize), &lmemsize, NULL);

                        cl_uint maxcus;
                        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxcus), &maxcus, NULL);

                        size_t maxwgsize;
                        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxwgsize), &maxwgsize, NULL);

                        log_info() << "OpenCL device: vendor = [" << vendor << "]";
                        log_info() << "OpenCL device: name = [" << name << "]";
                        log_info() << "OpenCL device: driver = [" << driver << "]";
                        log_info() << "OpenCL device: version = [" << version << "]";
                        log_info() << "OpenCL device: global memory size = " << gmemsize << " B";
                        log_info() << "OpenCL device: local memory size = " << lmemsize << " B";
                        log_info() << "OpenCL device: maximum compute units = " << maxcus;
                        log_info() << "OpenCL device: maximum work group size = " << maxwgsize;
                }

                cl_int err;
                rcontext_t context = make_shared(clCreateContext(NULL, 1, &device, NULL, NULL, &err));
                if (err != CL_SUCCESS)
                {
                        log_error() << "Failed to create an OpenCL GPU device!";
                        return rcontext_t();
                }

                return context;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rqueue_t opencl::make_command_queue(const rcontext_t& context, cl_device_id& device)
        {
                cl_int err;
                cl_device_id *devices;
                size_t deviceBufferSize = -1;

                // First get the size of the devices buffer
                err = clGetContextInfo(context.get(), CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
                if (err != CL_SUCCESS)
                {
                        throw std::runtime_error("Failed to query OpenCL context information!");
                }

                if (deviceBufferSize <= 0)
                {
                        throw std::runtime_error("No OpenCL devices available!");
                }

                // Allocate memory for the devices buffer
                devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
                err = clGetContextInfo(context.get(), CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
                if (err != CL_SUCCESS)
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
                assert(context);
                assert(device);

                const char* str = text.c_str();
                rprogram_t program = make_shared(clCreateProgramWithSource(context.get(), 1, (const char**)&str, NULL, NULL));
                if (!program)
                {
                        throw std::runtime_error("Failed to create OpenCL program from source!");
                }

                cl_int err = clBuildProgram(program.get(), 0, NULL, NULL, NULL, NULL);
                if (err != CL_SUCCESS)
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
                assert(program);

                return  make_shared(clCreateKernel(
                        program.get(), kname.c_str(), NULL));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rmem_t opencl::impl::make_read_mem(const rcontext_t& context, void* data, size_t mem_size)
        {
                assert(context);

                return  make_shared(clCreateBuffer(
                        context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, data, NULL));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opencl::rmem_t opencl::impl::make_write_mem(const rcontext_t& context, size_t mem_size)
        {
                assert(context);

                return  make_shared(clCreateBuffer(
                        context.get(), CL_MEM_READ_WRITE, mem_size, NULL, NULL));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool opencl::set_argument(const rkernel_t& kernel, size_t index, const rmem_t& mem)
        {
                assert(kernel);

                return clSetKernelArg(kernel.get(), index, sizeof(cl_mem), &mem) == CL_SUCCESS;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
