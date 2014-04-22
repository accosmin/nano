#include "zcompress.h"
#include <zlib.h>
#include <fstream>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        bool zuncompress(std::istream& istream, std::size_t bytes, std::vector<unsigned char>& data)
        {
                // zlib decompression buffers
                static const std::size_t CHUNK = 64 * 1024;

                z_stream strm;
                unsigned char in[CHUNK];
                unsigned char out[CHUNK];

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
                std::size_t num_bytes = bytes;
                for ( ; num_bytes > 0; )
                {
                        const std::size_t to_read = num_bytes >= CHUNK ? CHUNK : num_bytes;
                        num_bytes -= to_read;

                        if (!istream.read(reinterpret_cast<char*>(in), to_read))
                        {
                                inflateEnd(&strm);
                                return false;
                        }

                        strm.avail_in = to_read;
                        strm.next_in = in;

                        do
                        {
                                strm.avail_out = CHUNK;
                                strm.next_out = out;

                                const int ret = inflate(&strm, Z_NO_FLUSH);
                                if (ret != Z_OK && ret != Z_STREAM_END)
                                {
                                        inflateEnd(&strm);
                                        return false;
                                }

                                const std::size_t have = CHUNK - strm.avail_out;
                                data.insert(data.end(), out, out + have);
                        }
                        while (strm.avail_out == 0);
                }

                inflateEnd(&strm);

                // OK
                return true;
        }

	/////////////////////////////////////////////////////////////////////////////////////////
}
