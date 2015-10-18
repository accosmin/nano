#include "mstream.h"
#include <algorithm>

namespace cortex
{
        static bool isendl(char c)
        {
                return (c == '\n') || (c == '\r');
        }

        mstream_t::mstream_t(const char* data, std::size_t size)
                :       m_data(data),
                        m_size(size),
                        m_tellg(0),
                        m_gcount(0)
        {
        }

        bool mstream_t::read(char* bytes, std::size_t num_bytes)
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

        bool mstream_t::getline(std::string& line)
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

        bool mstream_t::skip(std::size_t num_bytes)
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

        std::size_t mstream_t::gcount() const
        {
                return m_gcount;
        }

        std::size_t mstream_t::tellg() const
        {
                return m_tellg;        
        }

        std::size_t mstream_t::size() const
        {
                return m_size;
        }

        mstream_t::operator bool() const
        {
                return tellg() < size();
        }
}
