#include "ncv.h"
#include "opencl/opencl.h"
#include <fstream>
#include <sstream>

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

        try
        {
                if (!ocl::manager_t::instance().valid())
                {
                        exit(EXIT_FAILURE);
                }

                const cl::Context& context = ocl::manager_t::instance().context();
                const cl::CommandQueue& queue = ocl::manager_t::instance().queue();

                const cl::Program program = ocl::manager_t::instance().program_from_text(program_source);

                cl::Kernel kernel = cl::Kernel(program, "add_kernel");

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
                cl::Buffer cl_a = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL);
                cl::Buffer cl_b = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL);
                //our output array
                cl::Buffer cl_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, array_size, NULL);

                cl::Event event;

                //push our CPU arrays to the GPU
                queue.enqueueWriteBuffer(cl_a, CL_TRUE, 0, array_size, a.data(), NULL, &event);
                ///clReleaseEvent(event); //we need to release events in order to be completely clean (has to do with openclprof)
                queue.enqueueWriteBuffer(cl_b, CL_TRUE, 0, array_size, b.data(), NULL, &event);
                ///clReleaseEvent(event);
                queue.enqueueWriteBuffer(cl_c, CL_TRUE, 0, array_size, c.data(), NULL, &event);
                ///clReleaseEvent(event);

                //set the arguements of our kernel
                kernel.setArg(0, cl_a);
                kernel.setArg(1, cl_b);
                kernel.setArg(2, cl_c);
                //Wait for the command queue to finish these commands before proceeding
                queue.finish();

                //for now we make the workgroup size the same as the number of elements in our arrays
                //workGroupSize[0] = num;

                //execute the kernel
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
                ///clReleaseEvent(event);
                ///clFinish(command_queue);
                queue.finish();

                //lets check our calculations by reading from the device memory and printing out the results
                queue.enqueueReadBuffer(cl_c, CL_TRUE, 0, array_size, c.data(), NULL, &event);
                //clReleaseEvent(event);

                for (size_t i = 0; i < num; i ++)
                {
                        log_info() << "result [" << (i + 1) << "/" << num << "]: " << c[i];
                }
        }

        catch (cl::Error e)
        {
                log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
