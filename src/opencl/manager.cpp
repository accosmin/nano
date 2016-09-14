#include "logger.h"
#include "kernels.h"
#include "manager.h"
#include <fstream>
#include <sstream>
#include <map>

namespace nano
{
        void opencl_manager_t::init()
        {
                const cl_int ret = cl::Platform::get(&m_platforms);
                if (m_platforms.empty() || ret != CL_SUCCESS)
                {
                        log_error() << "cannot find any OpenCL platform!";
                        throw cl::Error(ret);
                }

                for (size_t i = 0; i < m_platforms.size(); ++ i)
                {
                        const cl::Platform& platform = m_platforms[i];

                        std::stringstream ss;
                        ss << "OpenCL platform [" << (i + 1) << "/" << m_platforms.size() << "]: ";
                        const std::string base = ss.str();

                        log_info() << base << "CL_PLATFORM_NAME: " << platform.getInfo<CL_PLATFORM_NAME>();
                        log_info() << base << "CL_PLATFORM_VENDOR: " << platform.getInfo<CL_PLATFORM_NAME>();
                        log_info() << base << "CL_PLATFORM_VERSION: " << platform.getInfo<CL_PLATFORM_NAME>();

                        std::vector<cl::Device> devices;
                        const cl_int ret = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                        if (devices.empty() || ret != CL_SUCCESS)
                        {
                                log_error() << "cannot find any OpenCL device for the current platform!";
                                throw cl::Error(ret);
                        }

                        m_devices.insert(m_devices.end(), devices.begin(), devices.end());

                        for (size_t j = 0; j < devices.size(); j ++)
                        {
                                const cl::Device& device = devices[j];

                                const std::string name = device.getInfo<CL_DEVICE_NAME>();
                                const std::string vendor = device.getInfo<CL_DEVICE_VENDOR>();
                                const std::string driver = device.getInfo<CL_DRIVER_VERSION>();
                                const std::string version = device.getInfo<CL_DEVICE_VERSION>();
                                const std::string type = device_type_string(device.getInfo<CL_DEVICE_TYPE>());

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
                                ss << "-> OpenCL device [" << (j + 1) << "/" << devices.size() << "]: ";
                                const std::string base = ss.str();

                                log_info() << base << "CL_DEVICE_NAME: " << name;
                                log_info() << base << "CL_DEVICE_VENDOR:" << vendor;
                                log_info() << base << "CL_DRIVER_VERSION: " << driver;
                                log_info() << base << "CL_DEVICE_VERSION: " << version;
                                log_info() << base << "CL_DEVICE_TYPE: " << type;
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
        }

        bool opencl_manager_t::select(const cl_device_type type)
        {
                // <number of compute units, index>
                std::map<size_t, size_t, std::greater<size_t>> cuByIndex;
                for (size_t i = 0; i < m_devices.size(); ++ i)
                {
                        const cl::Device& device = m_devices[i];
                        cuByIndex[device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()] = i;
                }

                // select the device with the maximum number of compute units and the given type (if possible)
                for (const auto& cu : cuByIndex)
                {
                        const cl::Device& device = m_devices[cu.second];
                        if (    type == CL_DEVICE_TYPE_ALL ||
                                device.getInfo<CL_DEVICE_TYPE>() == type)
                        {
                                select(device);
                                return true;
                        }
                }

                if (!cuByIndex.empty())
                {
                        const auto& cu = *cuByIndex.begin();
                        select(m_devices[cu.second]);
                        return true;
                }
                else
                {
                        return false;
                }
        }

        void opencl_manager_t::select(const cl::Device& device)
        {
                log_info()
                        << "selected OpenCL device " << device.getInfo<CL_DEVICE_NAME>()
                        << " of type " << device_type_string(device.getInfo<CL_DEVICE_TYPE>()) << ".";

                m_device = device;
                m_context = cl::Context({m_device});
                m_command_queue = cl::CommandQueue(m_context, m_device, 0);
                m_program = make_program_from_text(opencl_kernels());
        }

        cl::Program opencl_manager_t::make_program_from_file(const std::string& filepath) const
        {
                std::ifstream file(filepath, std::ios::in);

                if (file.is_open())
                {
                        std::ostringstream oss;
                        oss << file.rdbuf();
                        return make_program_from_text(oss.str());
                }

                else
                {
                        return make_program_from_text("cannot load file!");
                }
        }

        cl::Program opencl_manager_t::make_program_from_text(const std::string& source) const
        {
                cl::Program::Sources sources(1, source);
                cl::Program program = cl::Program(m_context, sources);

                try
                {
                        program.build({m_device}, "-cl-mad-enable");//, "-cl-fast-relaxed-math");
                }
                catch (cl::Error& e)
                {
                        // load compilation errors
                        log_error() << "OpenCL program build status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(m_device);
                        log_error() << "OpenCL program build options: " << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(m_device);
                        log_error() << "OpenCL program build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);

                        // and re-throw the exception
                        throw cl::Error(e.err());
                }

                return program;
        }

        cl::Kernel opencl_manager_t::kernel(const char* name) const
        {
                return make_kernel(m_program, name);
        }

        cl::Kernel opencl_manager_t::make_kernel(const cl::Program& program, const char* name) const
        {
                try
                {
                        return cl::Kernel(program, name);
                }
                catch (cl::Error& e)
                {
                        // load kernel errors
                        log_error() << "OpenCL kernel error: " << error_string(e.err());

                        // and re-throw the exception
                        throw cl::Error(e.err());
                }
        }

        cl::Buffer opencl_manager_t::make_buffer(const size_t bytesize, cl_mem_flags flags) const
        {
                return cl::Buffer(m_context, flags, bytesize);
        }
}

