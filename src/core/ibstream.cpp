#include "ibstream.h"
#include <vector>

using namespace nano;

ibstream_t::ibstream_t(const std::string& path) :
        m_stream(path, std::ios::binary | std::ios::in)
{
}

std::streamsize ibstream_t::read(char* bytes, const std::streamsize num_bytes)
{
        m_stream.read(bytes, num_bytes);
        return m_stream.gcount();
}

bool ibstream_t::read(std::string& str)
{
        std::streamsize size = 0;
        if (read(size))
        {
                std::vector<char> buffer(static_cast<std::size_t>(size));
                if (read(buffer.data(), size) == size)
                {
                        str.assign(buffer.data(), static_cast<std::size_t>(size));
                        return true;
                }
        }
        return false;
}
