#include "io_gzip.h"
#include "io_stream.h"
#include <zlib.h>
#include <fstream>

#include <iostream>

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

                static const int windowBits = 15;
                static const int ENABLE_ZLIB_GZIP = 32;

                strm.zalloc = Z_NULL;
                strm.zfree = Z_NULL;
                strm.opaque = Z_NULL;
                strm.avail_in = 0;
                strm.next_in = Z_NULL;
                if (inflateInit2(&strm, windowBits | ENABLE_ZLIB_GZIP) != Z_OK)
                {
                        return false;
                }

                // decompress the data chunk
                while (num_bytes > 0 && istream)
                {
                        const std::streamsize to_read = (num_bytes >= chunk_size) ? chunk_size : num_bytes;
                        num_bytes -= to_read;

                        std::cout << "num_bytes = " << num_bytes << ", to_read = " << to_read << "\n";

                        if (!istream.read(reinterpret_cast<char*>(in), to_read))
                        {
                                std::cout << "eof = " << ", data.size() = " << data.size() << std::endl;
                                inflateEnd(&strm);
                                return false;
                        }

                        strm.avail_in = static_cast<uInt>(istream.gcount());
                        strm.next_in = in;

                        std::cout << "num_bytes = " << num_bytes << ", to_read = " << to_read << ", read = " << istream.gcount() << "\n";

                        do
                        {
                                strm.avail_out = static_cast<uInt>(chunk_size);
                                strm.next_out = out;

                                const int ret = inflate(&strm, Z_NO_FLUSH);
                                if (ret != Z_OK && ret != Z_STREAM_END)
                                {
                                        std::cout << "ret = " << ret << std::endl;
                                        inflateEnd(&strm);
                                        return false;
                                }

                                const std::streamsize have = chunk_size - strm.avail_out;
                                data.insert(data.end(), out, out + have);
                        }
                        while (strm.avail_out == 0);
                }

                inflateEnd(&strm);

                std::cout << "num_bytes = " << num_bytes << ", data.size() = " << data.size() << "\n";

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

        bool io::uncompress_gzip(const std::string& path, data_t& data)
        {
                std::ifstream istream(path.c_str(), std::ios_base::binary | std::ios_base::in);
                return istream.is_open() && uncompress_gzip(istream, data);
        }

        bool io::uncompress_gzip(const data_t& istream, data_t& data)
        {
                stream_t stream(istream);
                return io_uncompress_gzip(stream, stream.size(), data);
        }
}
