#include "imstream.h"
#include <algorithm>

namespace zob
{
        static bool isendl(char c)
        {
                return (c == '\n') || (c == '\r');
        }

        imstream_t& imstream_t::read(char* bytes, std::streamsize num_bytes)
        {
                if (tellg() + num_bytes >= size())
                {
                        num_bytes = size() - tellg();
                }

                std::copy(m_data + m_tellg, m_data + (m_tellg + num_bytes), bytes);

                m_tellg += num_bytes;
                m_gcount = num_bytes;

                return *this;
        }

        imstream_t& imstream_t::getline(std::string& line)
        {
                char c;

                for ( ; m_tellg < m_size && isendl(c = m_data[tellg()]); ++ m_tellg)
                {
                }

                line.clear();
                for ( ; m_tellg < m_size && !isendl(c = m_data[tellg()]); ++ m_tellg)
                {
                        line.push_back(c);
                }

                return *this;
        }

        imstream_t& imstream_t::seekg(std::streampos pos)
        {
                m_tellg = pos;
                m_gcount = 0;

                return *this;
        }

        std::streamsize imstream_t::gcount() const
        {
                return m_gcount;
        }

        std::streamsize imstream_t::tellg() const
        {
                return m_tellg;        
        }

        std::streamsize imstream_t::size() const
        {
                return m_size;
        }

        bool imstream_t::eof() const
        {
                return tellg() >= size();
        }

        bool imstream_t::good() const
        {
                return !eof();
        }

        imstream_t::operator bool() const
        {
                return gcount() > 0 || !eof();
        }

        const char* imstream_t::data() const
        {
                return m_data;
        }
}
