#include "istream.h"
#include <algorithm>

namespace nano
{
        static bool isendl(char c)
        {
                return (c == '\n') || (c == '\r');
        }

        istream_t::istream_t() :
                m_index(0),
                m_status(status::ok),
                m_tellg(0)
        {
        }

        bool istream_t::read(char* bytes, const std::streamsize num_bytes)
        {
                // read the missing buffered data (if possible)
                if (m_status == status::ok && m_index + num_bytes > static_cast<std::streamsize>(m_buffer.size()))
                {
                        m_status = advance(m_index + num_bytes, m_buffer);
                }

                if (    (m_status == status::ok || m_status == status::eof) &&
                        m_index + num_bytes <= static_cast<std::streamsize>(m_buffer.size()))
                {
                        // data buffer data to output
                        const char* data = m_buffer.data();
                        std::copy(data + m_index, data + (m_index + num_bytes), bytes);
                        m_index += num_bytes;
                        m_tellg += num_bytes;

                        // keep buffer small enough
                        const auto max_buffer_size = size_t(1024) * size_t(1024);
                        if (m_buffer.size() > max_buffer_size)
                        {
                                m_buffer.erase(m_buffer.begin(), m_buffer.begin() + m_index);
                                m_index = 0;
                        }
                        return true;
                }
                else
                {
                        return false;
                }
        }

        bool istream_t::getline(std::string& line)
        {
                char c;
                while (read(&c, 1) && isendl(c))
                {
                }

                line.clear();
                while (read(&c, 1) && !isendl(c))
                {
                        line.push_back(c);
                }

                return m_status != status::error && !line.empty();
        }

        std::streamsize istream_t::tellg() const
        {
                return m_tellg;
        }
}
