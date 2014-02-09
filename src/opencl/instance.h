#ifndef NANOCV_OPENCL_INSTANCE_H
#define NANOCV_OPENCL_INSTANCE_H

#include "cl_shared.h"
#include <string>

namespace ncv
{
        namespace opencl
        {
                ///
                /// \brief construct an OpenCL context
                ///
                rcontext_t make_context();

                ///
                /// \brief construct an OpenCL command queue on the given context
                ///
                rqueue_t make_command_queue(const rcontext_t&, cl_device_id&);

                ///
                /// \brief construct an OpenCL program on the given context from text
                ///
                rprogram_t make_program_from_text(const rcontext_t&, cl_device_id, const std::string& text);

                ///
                /// \brief construct an OpenCL program on the given context from source file
                ///
                rprogram_t make_program_from_file(const rcontext_t&, cl_device_id, const std::string& filepath);

                ///
                /// \brief construct an OpenCL kernel from the given program with the given name
                ///
                rkernel_t make_kernel(const rprogram_t& program, const std::string& name);

                rmem_t make_read_mem(const rcontext_t& context, float* data, size_t n_elements);
                rmem_t make_write_mem(const rcontext_t& context, size_t n_elements);
        }
}

#endif // NANOCV_OPENCL_INSTANCE_H
