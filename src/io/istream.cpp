#include "istream.h"
#include <algorithm>

namespace nano
{
        static bool isendl(char c)
        {
                return (c == '\n') || (c == '\r');
        }

        static size_t max_buffer_size()
        {
                return size_t(1024) * size_t(1024);
        }

        istream_t::istream_t() :
                m_index(0),
                m_status(io_status::good),
                m_tellg(0),
                m_gcount(0)
        {
        }

        std::streamsize istream_t::buffer(const std::streamsize num_bytes)
        {
                // read the missing buffered data (if possible)
                if (    m_status == io_status::good &&
                        available() < num_bytes)
                {
                        m_status = advance(m_index + num_bytes, m_buffer);
                }

                // return the number of bytes available
                return available();
        }

        void istream_t::advance(const std::streamsize num_bytes)
        {
                m_index += num_bytes;
                m_tellg += num_bytes;
        }

        void istream_t::trim()
        {
                // keep buffer small enough
                if (m_buffer.size() > max_buffer_size())
                {
                        m_buffer.erase(m_buffer.begin(), m_buffer.begin() + m_index);
                        m_index = 0;
                }
        }

        std::streamsize istream_t::read(char* bytes, const std::streamsize num_bytes)
        {
                m_gcount = std::min(buffer(num_bytes), num_bytes);
                std::copy(m_buffer.data() + m_index, m_buffer.data() + (m_index + m_gcount), bytes);
                advance(m_gcount);
                trim();
                return gcount();
        }

        std::streamsize istream_t::skip()
        {
                const auto num_bytes = std::streamsize(64 * 1024);
                auto read_bytes = std::streamsize(0);
                while ((read_bytes = buffer(num_bytes)) > 0 && m_status == io_status::good)
                {
                        advance(read_bytes);
                        trim();
                }
                return tellg();
        }

        bool istream_t::getline(std::string& line)
        {
                /// \todo not very efficient: should buffer larger chunks (1K ?!) and check for endline there!
                char c;
                while (read(&c, 1) && isendl(c)) {}

                line.clear();
                while (read(&c, 1) && !isendl(c)) { line.push_back(c); }

                return m_status != io_status::error && !line.empty();
        }

        std::streamsize istream_t::tellg() const
        {
                return m_tellg;
        }

        std::streamsize istream_t::available() const
        {
                return static_cast<std::streamsize>(m_buffer.size()) - m_index;
        }

        std::streamsize istream_t::gcount() const
        {
                return m_gcount;
        }
}
