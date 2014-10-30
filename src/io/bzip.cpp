#include "bzip.h"
#include "stream.h"
#include <bzlib.h>
#include <fstream>

namespace ncv
{
        using io::size_t;

        template
        <
                typename tstream
        >
        bool io_uncompress_bzip2(tstream& istream, size_t num_bytes, io::data_t& data)
        {
                // zlib decompression buffers
                static const std::streamsize chunk_size = 64 * 1024;

                bz_stream strm;
                char in[chunk_size];
                char out[chunk_size];

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
                while (num_bytes > 0 && istream)
                {
                        const std::streamsize to_read = (num_bytes >= chunk_size) ? chunk_size : num_bytes;
                        num_bytes -= to_read;

                        if (!istream.read(in, to_read))
                        {
                                BZ2_bzDecompressEnd(&strm);
                                return false;
                        }

                        strm.avail_in = static_cast<unsigned int>(istream.gcount());
                        strm.next_in = in;

                        do
                        {
                                strm.avail_out = static_cast<unsigned int>(chunk_size);
                                strm.next_out = out;

                                const int ret = BZ2_bzDecompress(&strm);
                                if (ret != BZ_OK && ret != BZ_STREAM_END)
                                {
                                        BZ2_bzDecompressEnd(&strm);
                                        return false;
                                }

                                const size_t have = chunk_size - strm.avail_out;
                                data.insert(data.end(), out, out + have);
                        }
                        while (strm.avail_out == 0);
                }

                BZ2_bzDecompressEnd(&strm);

                // OK
                return (num_bytes == std::string::npos) ? true : (num_bytes == 0);
        }

        bool io::uncompress_bzip2(std::istream& istream, size_t num_bytes, data_t& data)
        {
                return io_uncompress_bzip2(istream, num_bytes, data);
        }

        bool io::uncompress_bzip2(std::istream& istream, data_t& data)
        {
                return uncompress_bzip2(istream, std::string::npos, data);
        }

        bool io::uncompress_bzip2(const data_t& istream, data_t& data)
        {
                stream_t stream(istream.data(), istream.size());
                return io_uncompress_bzip2(stream, stream.size(), data);
        }
}
