#include "gzip.h"
#include "stream.h"
#include <zlib.h>
#include <fstream>

namespace ncv
{
        using io::size_t;

        template
        <
                typename tstream
        >
        bool io_uncompress_gzip(tstream& istream, size_t num_bytes, io::data_t& data)
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

                // decompress the data chunk
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
                                data.insert(data.end(), out, out + have);
                        }
                        while (strm.avail_out == 0);
                }

                inflateEnd(&strm);

                // OK
                return (num_bytes == std::string::npos) ? true : (num_bytes == 0);
        }

        bool io::uncompress_gzip(std::istream& istream, size_t num_bytes, data_t& data)
        {
                return io_uncompress_gzip(istream, num_bytes, data);
        }

        bool io::uncompress_gzip(std::istream& istream, data_t& data)
        {
                return uncompress_gzip(istream, std::string::npos, data);
        }

        bool io::uncompress_gzip(const data_t& istream, data_t& data)
        {
                stream_t stream(istream.data(), istream.size());
                return io_uncompress_gzip(stream, stream.size(), data);
        }
}
