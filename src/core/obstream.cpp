#include "obstream.h"

using namespace nano;

obstream_t::obstream_t(const std::string& path) :
        m_stream(path, std::ios::binary | std::ios::out | std::ios::trunc)
{
}

bool obstream_t::write(const char* bytes, const std::streamsize num_bytes)
{
        return m_stream.write(bytes, num_bytes).good();
}

bool obstream_t::write(const std::string& str)
{
        return  write(static_cast<std::streamsize>(str.size())) &&
                write(str.data(), static_cast<std::streamsize>(str.size()));
}
