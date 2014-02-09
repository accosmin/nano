#include "ncv.h"
#include "opencl/opencl.h"
#include <iostream>
#include <fstream>
#include <sstream>

///
//  Constants
//
const int ARRAY_SIZE = 1000;

const char* program_source = "\n" \
"__kernel void add_kernel(                              \n" \
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

        std::vector<cl::Platform> platforms;
        cl_int err = cl::Platform::get(&platforms);
        log_info() << "OpenCL status (platform query): " << ncv::error_string(err);

        if (platforms.empty())
        {
                log_error() << "Cannot find any OpenCL platforms!";
                exit(EXIT_FAILURE);
        }

        cl_context_properties properties[] =
        {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)(platforms[0])(),
                0
        };
        cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        if (devices.empty())
        {
                log_error() << "Cannot find any OpenCL GPU device!";
                exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < devices.size(); i ++)
        {
                const cl::Device& device = devices[i];

                const std::string name = device.getInfo<CL_DEVICE_NAME>();
                const std::string vendor = device.getInfo<CL_DEVICE_VENDOR>();
                const std::string driver = device.getInfo<CL_DRIVER_VERSION>();
                const std::string version = device.getInfo<CL_DEVICE_VERSION>();

                const cl_ulong gmemsize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                const cl_ulong lmemsize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
                const cl_uint maxcus = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                const size_t maxwgsize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

                const size_t size = devices.size();
                log_info() << "OpenCL device [" << (i + 1) << "/" << size << "]: vendor = " << vendor << "";
                log_info() << "OpenCL device [" << (i + 1) << "/" << size << "]: name = " << name << "";
                log_info() << "OpenCL device [" << (i + 1) << "/" << size << "]: driver = " << driver << "";
                log_info() << "OpenCL device [" << (i + 1) << "/" << size << "]: version = " << version << "";
                log_info() << "OpenCL device [" << (i + 1) << "/" << size << "]: global memory size = " << gmemsize << " B";
                log_info() << "OpenCL device [" << (i + 1) << "/" << size << "]: local memory size = " << lmemsize << " B";
                log_info() << "OpenCL device [" << (i + 1) << "/" << size << "]: maximum compute units = " << maxcus;
                log_info() << "OpenCL device [" << (i + 1) << "/" << size << "]: maximum work group size = " << maxwgsize;
        }

        try
        {
                const cl::Device& device = devices[0];
                cl::CommandQueue queue = cl::CommandQueue(context, device, 0, &err);

                const std::string kernel_source = program_source;

                cl::Program::Sources source(1, std::make_pair(kernel_source.c_str(), kernel_source.size()));
                cl::Program program = cl::Program(context, source);
                err = program.build(devices);

                log_info() << "OpenCL status (program): " << ncv::error_string(err);

//                log_info() << "OpenCL program build status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
//                log_info() << "OpenCL program build options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device);
//                log_info() << "OpenCL program build log:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

                cl::Kernel kernel = cl::Kernel(program, "add_kernel", &err);

                log_info() << "OpenCL status (kernel): " << ncv::error_string(err);

                // initialize our CPU memory arrays, send them to the device and set the kernel arguements
                const size_t num = 16;

                std::vector<float> a(num);
                std::vector<float> b(num);
                std::vector<float> c(num);

                for (size_t i = 0; i < num; i ++)
                {
                        a[i] = 1.0f * i;
                        b[i] = 1.0f * i;
                        c[i] = 0.0f;
                }

                const size_t array_size = sizeof(float) * num;
                //our input arrays
                cl::Buffer cl_a = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL, &err);
                cl::Buffer cl_b = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL, &err);
                //our output array
                cl::Buffer cl_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, array_size, NULL, &err);

                cl::Event event;

                //push our CPU arrays to the GPU
                err = queue.enqueueWriteBuffer(cl_a, CL_TRUE, 0, array_size, a.data(), NULL, &event);
                ///clReleaseEvent(event); //we need to release events in order to be completely clean (has to do with openclprof)
                err = queue.enqueueWriteBuffer(cl_b, CL_TRUE, 0, array_size, b.data(), NULL, &event);
                ///clReleaseEvent(event);
                err = queue.enqueueWriteBuffer(cl_c, CL_TRUE, 0, array_size, c.data(), NULL, &event);
                ///clReleaseEvent(event);

                //set the arguements of our kernel
                err = kernel.setArg(0, cl_a);
                err = kernel.setArg(1, cl_b);
                err = kernel.setArg(2, cl_c);
                //Wait for the command queue to finish these commands before proceeding
                queue.finish();

                //for now we make the workgroup size the same as the number of elements in our arrays
                //workGroupSize[0] = num;

                //execute the kernel
                err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
                ///clReleaseEvent(event);
                log_info() << "OpenCL status (run kernel): " << ncv::error_string(err);
                ///clFinish(command_queue);
                queue.finish();

                //lets check our calculations by reading from the device memory and printing out the results
                err = queue.enqueueReadBuffer(cl_c, CL_TRUE, 0, array_size, c.data(), NULL, &event);
                log_info() << "OpenCL status (read buffer): " << ncv::error_string(err);

                //clReleaseEvent(event);

                for (size_t i = 0; i < num; i ++)
                {
                        log_info() << "result [" << (i + 1) << "/" << num << "]: " << c[i];
                }
        }
        catch (cl::Error e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << e.err() << ")!";
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
