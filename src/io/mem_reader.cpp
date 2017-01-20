#include "mem_reader.h"

namespace nano
{
        io_status mem_reader_t::advance(const std::streamsize num_bytes, buffer_t& buffer)
        {
                const auto max_read = static_cast<size_t>(num_bytes);
                const auto to_read = (m_index + max_read > m_size) ? (m_size - m_index) : max_read;

                buffer.insert(buffer.end(), m_data + m_index, m_data + (m_index + to_read));
                m_index += to_read;

                return (m_index >= m_size) ? io_status::eof : io_status::ok;
        }
}
