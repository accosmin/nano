#include "istream_std.h"
#include <istream>

namespace nano
{
        std_istream_t::std_istream_t(std::istream& stream) :
                m_stream(stream)
        {
        }

        io_status std_istream_t::advance(const std::streamsize num_bytes, buffer_t& buffer)
        {
                static const auto buff_size = std::streamsize(64 * 1024);
                static char buff[buff_size];
                while ( static_cast<std::streamsize>(buffer.size()) < num_bytes &&
                        m_stream)
                {
                        m_stream.read(buff, buff_size);
                        buffer.insert(buffer.end(), buff, buff + m_stream.gcount());
                }

                return  m_stream.good() ? io_status::good : (m_stream.eof() ? io_status::eof : io_status::error);
        }
}
