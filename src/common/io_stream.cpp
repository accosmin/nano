#include "io_stream.h"
#include <algorithm>

namespace ncv
{
        io::stream_t::stream_t(const data_t& data)
                :       m_data(data),
                        m_tellg(0),
                        m_gcount(0)
        {
        }

        bool io::stream_t::read(char* bytes, size_t num_bytes)
        {
                const size_t rem_bytes = remg();

                if (rem_bytes >= num_bytes)
                {
                        std::copy(m_data.data() + m_tellg, m_data.data() + (m_tellg + num_bytes), bytes);

                        m_tellg += num_bytes;
                        m_gcount = num_bytes;
                        return true;
                }
                else
                {
                        return false;
                }
        }

        bool io::stream_t::skip(size_t num_bytes)
        {
                const size_t rem_bytes = remg();

                if (rem_bytes >= num_bytes)
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

        size_t io::stream_t::remg() const
        {
                return (tellg() >= size()) ? 0 : (size() - tellg());
        }

        size_t io::stream_t::size() const
        {
                return m_data.size();
        }

        io::stream_t::operator bool() const
        {
                return tellg() < size();
        }
}
