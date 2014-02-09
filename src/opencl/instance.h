#ifndef NANOCV_OPENCL_INSTANCE_H
#define NANOCV_OPENCL_INSTANCE_H

#include "cl_shared.h"
#include <string>
#include "util/singleton.hpp"

namespace ncv
{
        namespace opencl
        {
                struct id_t
                {
                        id_t(size_t id = 0)
                                :       m_id(id)
                        {
                        }

                        operator bool() const
                        {
                                return m_id != 0;
                        }

                        void operator++()
                        {
                                m_id ++;
                        }

                        size_t          m_id;
                };

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

                namespace impl
                {
                        rmem_t make_read_mem(const rcontext_t& context, void* data, size_t mem_size);
                        rmem_t make_write_mem(const rcontext_t& context, size_t mem_size);
                }

                template
                <
                        typename tscalar
                >
                rmem_t make_read_mem(const rcontext_t& context, tscalar* data, size_t n_elements)
                {
                        return impl::make_read_mem(context, (void*)data, sizeof(tscalar) * n_elements);
                }

                template
                <
                        typename tscalar
                >
                rmem_t make_write_mem(const rcontext_t& context, tscalar*, size_t n_elements)
                {
                        return impl::make_write_mem(context, sizeof(tscalar) * n_elements);
                }

                bool set_argument(const rkernel_t& kernel, size_t index, const rmem_t& mem);
        }
}

#endif // NANOCV_OPENCL_INSTANCE_H
