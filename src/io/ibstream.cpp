#include "ibstream.h"
#include <vector>
#include <istream>

namespace io
{
        ibstream_t::ibstream_t(std::istream& stream)
                :       m_stream(stream)
        {
                m_stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);
        }

        ibstream_t& ibstream_t::read(std::string& str)
        {
                std::vector<char> buffer;
                read(buffer);

                str.resize(buffer.size());
                str.assign(buffer.data(), buffer.size());

                return *this;
        }

        ibstream_t& ibstream_t::read_blob(char* data, const std::size_t count)
        {
                m_stream.read(data, static_cast<std::streamsize>(count));
                return *this;
        }
}
