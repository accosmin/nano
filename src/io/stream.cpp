#include "stream.h"
#include <algorithm>

namespace ncv
{
        static bool isendl(char c)
        {
                return (c == '\n') || (c == '\r');
        }

        io::stream_t::stream_t(const char* data, size_t size)
                :       m_data(data),
                        m_size(size),
                        m_tellg(0),
                        m_gcount(0)
        {
        }

        bool io::stream_t::read(char* bytes, size_t num_bytes)
        {
                if (tellg() + num_bytes <= size())
                {
                        std::copy(m_data + m_tellg, m_data + (m_tellg + num_bytes), bytes);

                        m_tellg += num_bytes;
                        m_gcount = num_bytes;
                        return true;
                }
                else
                {
                        return false;
                }
        }

        bool io::stream_t::getline(std::string& line)
        {
                char c;

                for ( ; m_tellg < m_size && isendl(c = m_data[tellg()]); m_tellg ++)
                {
                }

                line.clear();
                for ( ; m_tellg < m_size && !isendl(c = m_data[tellg()]); m_tellg ++)
                {
                        line.push_back(c);
                }

                return !line.empty();
        }

        bool io::stream_t::skip(size_t num_bytes)
        {
                if (tellg() + num_bytes <= size())
                {
                        m_tellg += num_bytes;
                        m_gcount = 0;
                        return true;
                }
                else
                {
                        return false;
                }
        }

        size_t io::stream_t::gcount() const
        {
                return m_gcount;
        }

        size_t io::stream_t::tellg() const
        {
                return m_tellg;        
        }

        size_t io::stream_t::size() const
        {
                return m_size;
        }

        io::stream_t::operator bool() const
        {
                return tellg() < size();
        }
}
