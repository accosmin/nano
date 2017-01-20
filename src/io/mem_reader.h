#pragma once

#include "buffer.h"

namespace nano
{
        ///
        /// \brief streaming of binary data using an in-memory buffer.
        ///
        class NANO_PUBLIC mem_reader_t
        {
        public:

                template <typename tsize>
                mem_reader_t(const char* data, const tsize size) :
                        m_data(data),
                        m_size(static_cast<std::size_t>(size)),
                        m_index(0)
                {
                }

                mem_reader_t(const mem_reader_t&) = delete;
                mem_reader_t& operator=(const mem_reader_t&) = delete;

                ~mem_reader_t() = default;

                io_status advance(const std::streamsize num_bytes, buffer_t& buffer);

        private:

                const char* const       m_data;
                const std::size_t       m_size;
                std::size_t             m_index;
        };
}
