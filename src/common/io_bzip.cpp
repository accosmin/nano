#include "io_bzip.h"
#include <bzlib.h>
#include <fstream>

namespace ncv
{
        bool io::uncompress_bzip2(std::istream& istream, std::size_t bytes, data_t& data)
        {
                // zlib decompression buffers
                static const std::size_t CHUNK = 64 * 1024;

                bz_stream strm;
                char in[CHUNK];
                char out[CHUNK];

                strm.bzalloc = NULL;
                strm.bzfree = NULL;
                strm.opaque = NULL;
                strm.avail_in = 0;
                strm.next_in = NULL;
                if (BZ2_bzDecompressInit(&strm, 0, 0) != BZ_OK)
                {
                        return false;
                }

                // decompress the data chunk
                std::size_t num_bytes = bytes;
                for ( ; num_bytes > 0; )
                {
                        const std::size_t to_read = num_bytes >= CHUNK ? CHUNK : num_bytes;
                        num_bytes -= to_read;

                        if (!istream.read(in, to_read))
                        {
                                BZ2_bzDecompressEnd(&strm);
                                return false;
                        }

                        strm.avail_in = to_read;
                        strm.next_in = in;

                        do
                        {
                                strm.avail_out = CHUNK;
                                strm.next_out = out;

                                const int ret = BZ2_bzDecompress(&strm);
                                if (ret != BZ_OK && ret != BZ_STREAM_END)
                                {
                                        BZ2_bzDecompressEnd(&strm);
                                        return false;
                                }

                                const std::size_t have = CHUNK - strm.avail_out;
                                data.insert(data.end(), out, out + have);
                        }
                        while (strm.avail_out == 0);
                }

                BZ2_bzDecompressEnd(&strm);

                // OK
                return num_bytes == 0;
        }
}
