#include "io_zlib.h"
#include <zlib.h>
#include <fstream>

namespace ncv
{
        bool io::uncompress_zlib(std::istream& istream, std::size_t bytes, data_t& data)
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
                while (bytes > 0 && istream)
                {
                        const std::streamsize to_read = bytes >= chunk_size ? chunk_size : bytes;
                        bytes -= to_read;

                        if (!istream.read(reinterpret_cast<char*>(in), to_read))
                        {
                                inflateEnd(&strm);
                                return false;
                        }

                        strm.avail_in = istream.gcount();
                        strm.next_in = in;

                        do
                        {
                                strm.avail_out = chunk_size;
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
                return (bytes == std::string::npos) ? true : (bytes == 0);
        }

        bool io::uncompress_zlib(std::istream& istream, data_t& data)
        {
                return uncompress_zlib(istream, std::string::npos, data);
        }

        bool io::uncompress_zlib(const std::string& path, data_t& data)
        {
                std::ifstream in(path.c_str(), std::ios_base::binary | std::ios_base::in);

                return  in.is_open() &&
                        uncompress_zlib(in, data);
        }
}
