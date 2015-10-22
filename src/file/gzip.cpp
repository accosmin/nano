#include "gzip.h"
#include <zlib.h>
#include <fstream>

namespace cortex
{
        template
        <
                typename tstream
        >
        bool io_uncompress_gzip(tstream& istream, const std::streamsize orig_num_bytes, buffer_t& buffer)
        {
                // zlib decompression buffers
                static const std::streamsize chunk_size = 64 * 1024;

                z_stream strm;
                unsigned char in[chunk_size];
                unsigned char out[chunk_size];

                strm.zalloc = Z_NULL;
                strm.zfree = Z_NULL;
                strm.opaque = Z_NULL;
                strm.avail_in = 0;
                strm.next_in = Z_NULL;
                if (inflateInit(&strm) != Z_OK)
                {
                        return false;
                }

                // decompress the buffer chunk
                std::streamsize num_bytes = orig_num_bytes;
                while (num_bytes > 0 && istream)
                {
                        const std::streamsize to_read = (num_bytes >= chunk_size) ? chunk_size : num_bytes;
                        num_bytes -= to_read;

                        if (!istream.read(reinterpret_cast<char*>(in), to_read))
                        {
                                inflateEnd(&strm);
                                return false;
                        }

                        strm.avail_in = static_cast<uInt>(istream.gcount());
                        strm.next_in = in;

                        do
                        {
                                strm.avail_out = static_cast<uInt>(chunk_size);
                                strm.next_out = out;

                                const int ret = inflate(&strm, Z_NO_FLUSH);
                                if (ret != Z_OK && ret != Z_STREAM_END)
                                {
                                        inflateEnd(&strm);
                                        return false;
                                }

                                const std::streamsize have = chunk_size - strm.avail_out;
                                buffer.insert(buffer.end(), out, out + have);
                        }
                        while (strm.avail_out == 0);
                }

                inflateEnd(&strm);

                // OK
                return (num_bytes == max_streamsize()) ? true : (num_bytes == 0);
        }

        bool uncompress_gzip(std::istream& istream, std::streamsize num_bytes, buffer_t& buffer)
        {
                return io_uncompress_gzip(istream, num_bytes, buffer);
        }

        bool uncompress_gzip(std::istream& istream, buffer_t& buffer)
        {
                return uncompress_gzip(istream, max_streamsize(), buffer);
        }

        bool uncompress_gzip(mstream_t& istream, buffer_t& buffer)
        {
                return io_uncompress_gzip(istream, istream.size(), buffer);
        }
}
