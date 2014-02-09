#ifndef NANOCV_OPENCL_H
#define NANOCV_OPENCL_H

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif

namespace ncv
{
        const char* error_string(cl_int error);

        std::string load_text_file(const std::string& filepath);
}

#endif // NANOCV_OPENCL_H
