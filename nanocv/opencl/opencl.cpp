#include "opencl.h"
#include "../logger.h"
#include <fstream>
#include <sstream>
#include <cassert>

namespace ncv
{
        ocl::manager_t& ocl::get_manager()
        {
                return ocl::manager_t::instance();
        }

        ocl::manager_t::manager_t()
        {
                try
                {
                        cl_int err = cl::Platform::get(&m_platforms);
                        log_info() << "OpenCL status (platform query): " << ocl::error_string(err);

                        if (m_platforms.empty())
                        {
                                log_error() << "Cannot find any OpenCL platforms!";
                                return;
                        }

                        cl_context_properties properties[] =
                        {
                                CL_CONTEXT_PLATFORM,
                                (cl_context_properties)(m_platforms[0])(),
                                0
                        };
                        const cl::Context context(CL_DEVICE_TYPE_GPU, properties);
                        m_devices = context.getInfo<CL_CONTEXT_DEVICES>();

                        if (m_devices.empty())
                        {
                                log_error() << "Cannot find any OpenCL GPU device!";
                                return;
                        }

                        for (size_t i = 0; i < m_devices.size(); i ++)
                        {
                                const cl::Device& device = m_devices[i];

                                const std::string name = device.getInfo<CL_DEVICE_NAME>();
                                const std::string vendor = device.getInfo<CL_DEVICE_VENDOR>();
                                const std::string driver = device.getInfo<CL_DRIVER_VERSION>();
                                const std::string version = device.getInfo<CL_DEVICE_VERSION>();

                                const size_t gmemsize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                                const size_t lmemsize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
                                const size_t amemsize = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
                                const size_t cmemsize = device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
                                const size_t maxcus = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                                const size_t maxwgsize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
                                const size_t maxwidims = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
                                const std::vector<size_t> maxwisizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
                                const size_t maxkparams = device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();

                                std::stringstream ss;
                                ss << "OpenCL device [" << (i + 1) << "/" << m_devices.size() << "]: ";
                                const std::string base = ss.str();

                                log_info() << base << "CL_DEVICE_NAME: " << name;
                                log_info() << base << "CL_DEVICE_VENDOR:" << vendor;
                                log_info() << base << "CL_DRIVER_VERSION: " << driver;
                                log_info() << base << "CL_DEVICE_VERSION: " << version;
                                log_info() << base << "CL_DEVICE_TYPE: " << "CL_DEVICE_TYPE_GPU";
                                log_info() << base << "CL_DEVICE_GLOBAL_MEM_SIZE: " << gmemsize << "B = "
                                           << (gmemsize / 1024) << "KB = " << (gmemsize / 1024 / 1024) << "MB";
                                log_info() << base << "CL_DEVICE_LOCAL_MEM_SIZE: " << lmemsize << "B = "
                                           << (lmemsize / 1024) << "KB = " << (lmemsize / 1024 / 1024) << "MB";
                                log_info() << base << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << cmemsize << "B = "
                                           << (cmemsize / 1024) << "KB = " << (cmemsize / 1024 / 1024) << "MB";
                                log_info() << base << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << amemsize << "B = "
                                           << (amemsize / 1024) << "KB = " << (amemsize / 1024 / 1024) << "MB";
                                log_info() << base << "CL_DEVICE_MAX_COMPUTE_UNITS: " << maxcus;
                                log_info() << base << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << maxwgsize;
                                log_info() << base << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << maxwidims;
                                log_info() << base << "CL_DEVICE_MAX_WORK_ITEM_SIZES: "
                                           << (maxwisizes.size() > 0 ? maxwisizes[0] : 0) << " / "
                                           << (maxwisizes.size() > 1 ? maxwisizes[1] : 0) << " / "
                                           << (maxwisizes.size() > 2 ? maxwisizes[2] : 0);
                                log_info() << base << "CL_DEVICE_MAX_PARAMETER_SIZE: " << maxkparams;
                        }
                }

                catch (cl::Error& e)
                {
                        log_error() << "OpenCL fatal error: <" << e.what() << "> (" << ocl::error_string(e.err()) << ")!";

                        m_platforms.clear();
                        m_devices.clear();
                }
        }

        cl::Context ocl::manager_t::make_context() const
        {
                assert(valid());

                cl_context_properties properties[] =
                {
                        CL_CONTEXT_PLATFORM,
                        (cl_context_properties)(m_platforms[0])(),
                        0
                };

                return cl::Context(CL_DEVICE_TYPE_GPU, properties);
        }

        cl::CommandQueue ocl::manager_t::make_command_queue(const cl::Context& context) const
        {
                assert(valid());

                const cl::Device& device = m_devices[0];
                return cl::CommandQueue(context, device, 0);
        }

        cl::Program ocl::manager_t::make_program_from_file(const cl::Context& context, const std::string& filepath) const
        {
                return make_program_from_text(context, ocl::load_text_file(filepath));
        }

        cl::Program ocl::manager_t::make_program_from_text(const cl::Context& context, const std::string& source) const
        {
                assert(valid());

                cl::Program::Sources sources(1, std::make_pair(source.c_str(), source.size()));
                cl::Program program = cl::Program(context, sources);

                try
                {
                        program.build(m_devices, "-cl-mad-enable");//, "-cl-fast-relaxed-math");
                }
                catch (cl::Error& e)
                {
                        // load compilation errors
                        const cl::Device& device = m_devices[0];
                        log_error() << "OpenCL program build status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                        log_error() << "OpenCL program build options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device);
                        log_error() << "OpenCL program build log:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

                        // and re-throw the exception
                        throw cl::Error(e.err());
                }

                return program;
        }

        cl::Kernel ocl::manager_t::make_kernel(const cl::Program& program, const std::string& name) const
        {
                assert(valid());

                return cl::Kernel(program, name.c_str());
        }

        cl::Buffer ocl::manager_t::make_buffer(const cl::Context& context, size_t bytesize, int flags) const
        {
                assert(valid());

                return cl::Buffer(context, flags, bytesize, NULL);
        }

        const char* ocl::error_string(cl_int error)
        {
                static const char* errorString[] =
                {
                        "CL_SUCCESS",
                        "CL_DEVICE_NOT_FOUND",
                        "CL_DEVICE_NOT_AVAILABLE",
                        "CL_COMPILER_NOT_AVAILABLE",
                        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
                        "CL_OUT_OF_RESOURCES",
                        "CL_OUT_OF_HOST_MEMORY",
                        "CL_PROFILING_INFO_NOT_AVAILABLE",
                        "CL_MEM_COPY_OVERLAP",
                        "CL_IMAGE_FORMAT_MISMATCH",
                        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
                        "CL_BUILD_PROGRAM_FAILURE",
                        "CL_MAP_FAILURE",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "CL_INVALID_VALUE",
                        "CL_INVALID_DEVICE_TYPE",
                        "CL_INVALID_PLATFORM",
                        "CL_INVALID_DEVICE",
                        "CL_INVALID_CONTEXT",
                        "CL_INVALID_QUEUE_PROPERTIES",
                        "CL_INVALID_COMMAND_QUEUE",
                        "CL_INVALID_HOST_PTR",
                        "CL_INVALID_MEM_OBJECT",
                        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
                        "CL_INVALID_IMAGE_SIZE",
                        "CL_INVALID_SAMPLER",
                        "CL_INVALID_BINARY",
                        "CL_INVALID_BUILD_OPTIONS",
                        "CL_INVALID_PROGRAM",
                        "CL_INVALID_PROGRAM_EXECUTABLE",
                        "CL_INVALID_KERNEL_NAME",
                        "CL_INVALID_KERNEL_DEFINITION",
                        "CL_INVALID_KERNEL",
                        "CL_INVALID_ARG_INDEX",
                        "CL_INVALID_ARG_VALUE",
                        "CL_INVALID_ARG_SIZE",
                        "CL_INVALID_KERNEL_ARGS",
                        "CL_INVALID_WORK_DIMENSION",
                        "CL_INVALID_WORK_GROUP_SIZE",
                        "CL_INVALID_WORK_ITEM_SIZE",
                        "CL_INVALID_GLOBAL_OFFSET",
                        "CL_INVALID_EVENT_WAIT_LIST",
                        "CL_INVALID_EVENT",
                        "CL_INVALID_OPERATION",
                        "CL_INVALID_GL_OBJECT",
                        "CL_INVALID_BUFFER_SIZE",
                        "CL_INVALID_MIP_LEVEL",
                        "CL_INVALID_GLOBAL_WORK_SIZE",
                };

                static const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

                const int index = -error;
                return (index >= 0 && index < errorCount) ? errorString[index] : "";
        }

        std::string ocl::load_text_file(const std::string& filepath)
        {
                std::ifstream file(filepath, std::ios::in);

                if (file.is_open())
                {
                        std::ostringstream oss;
                        oss << file.rdbuf();
                        return oss.str();
                }

                else
                {
                        return std::string();
                }
        }
}
	
