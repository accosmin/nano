#include "io.h"
#include <limits>
#include <fstream>

using namespace nano;

std::streamsize nano::max_streamsize()
{
        return std::numeric_limits<std::streamsize>::max();
}

bool nano::save_buffer(const std::string& path, const buffer_t& buffer)
{
        std::ofstream stream(path.c_str(), std::ios::binary | std::ios::out);
        if (!stream.is_open())
        {
                return false;
        }

        const std::streamsize stream_size = 64 * 1024;
        std::streamsize num_bytes = static_cast<std::streamsize>(buffer.size());

        const char* pbuffer = buffer.data();
        while (stream && num_bytes > 0)
        {
                const auto to_write = num_bytes >= stream_size ? stream_size : num_bytes;

                if (stream.write(pbuffer, to_write))
                {
                        pbuffer += to_write;
                        num_bytes -= to_write;
                }
        }

        return static_cast<bool>(stream);
}

bool nano::load_buffer(const std::string& path, buffer_t& buffer)
{
        std::ifstream stream(path.c_str(), std::ios::binary | std::ios::in);
        if (!stream.is_open())
        {
                return false;
        }

        const std::streamsize stream_size = 64 * 1024;
        char buff[stream_size];

        while (stream)
        {
                stream.read(buff, stream_size);
                buffer.insert(buffer.end(), buff, buff + stream.gcount());
        }

        return stream.eof();
}
