#include "buffer.h"
#include <istream>

namespace cortex
{
        bool load_buffer(std::istream& stream, std::streamsize num_bytes, buffer_t& buffer)
        {
                buffer.resize(static_cast<std::size_t>(num_bytes));
                const std::streamsize stream_size = 64 * 1024;

                char* pbuffer = buffer.data();
                while (stream && num_bytes > 0)
                {
                        const auto to_read = num_bytes >= stream_size ? stream_size : num_bytes;

                        if (stream.read(pbuffer, to_read))
                        {
                                pbuffer += to_read;
                                num_bytes -= to_read;
                        }
                }

                return num_bytes == 0;
        }
}
