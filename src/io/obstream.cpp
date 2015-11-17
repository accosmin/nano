#include "obstream.h"
#include <ostream>

namespace io
{
        obstream_t::obstream_t(std::ostream& stream)
                :       m_stream(stream)
        {
                m_stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);
        }

        obstream_t& obstream_t::write(const std::string& str)
        {
                write(str.size());
                return write_blob(str.data(), str.size());
        }

        obstream_t& obstream_t::write_blob(const char* data, const std::size_t count)
        {
                m_stream.write(data, static_cast<std::streamsize>(count));
                return *this;
        }
}
