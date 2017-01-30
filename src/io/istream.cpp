#include "istream.h"
#include <cassert>
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

        static std::streamsize chunk_size()
        {
                return 1024 * 1024;
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

        void istream_t::advance()
        {
                m_index += m_gcount;
                m_tellg += m_gcount;
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
                if (bytes)
                {
                       std::copy(m_buffer.data() + m_index, m_buffer.data() + (m_index + m_gcount), bytes);
                }
                advance();
                trim();
                return gcount();
        }

        std::streamsize istream_t::skip()
        {
                while (*this)
                {
                        read(nullptr, chunk_size());
                }
                return tellg();
        }

        bool istream_t::skip(std::streamsize num_bytes)
        {
                while (num_bytes > 0 && (*this))
                {
                        const auto read_bytes = read(nullptr, std::min(num_bytes, chunk_size()));
                        num_bytes -= read_bytes;
                }
                return num_bytes == 0;
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
                assert(m_index <= static_cast<std::streamsize>(m_buffer.size()));
                return static_cast<std::streamsize>(m_buffer.size()) - m_index;
        }

        std::streamsize istream_t::gcount() const
        {
                return m_gcount;
        }

        istream_t::operator bool() const
        {
                return (m_status == io_status::good) || (m_status == io_status::eof && available() > 0);
        }
}
