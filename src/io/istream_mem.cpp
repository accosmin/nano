#include "istream_mem.h"

namespace nano
{
        io_status mem_istream_t::advance(const std::streamsize num_bytes, buffer_t& buffer)
        {
                const auto to_read = (m_index + num_bytes > m_size) ? (m_size - m_index) : num_bytes;

                buffer.insert(buffer.end(), m_data + m_index, m_data + (m_index + to_read));
                m_index += to_read;

                return (m_index >= m_size) ? io_status::eof : io_status::good;
        }
}
