#include "istream_zlib.h"

using namespace nano;

zlib_istream_t::zlib_istream_t(istream_t& istream, const std::streamsize max_num_bytes) :
        m_istream(istream),
        m_max_num_bytes(max_num_bytes)
{
        m_zstream.zalloc = Z_NULL;
        m_zstream.zfree = Z_NULL;
        m_zstream.opaque = Z_NULL;
        m_zstream.avail_in = 0;
        m_zstream.next_in = Z_NULL;
        inflateInit(&m_zstream);
}

zlib_istream_t::~zlib_istream_t()
{
        inflateEnd(&m_zstream);
}

io_status zlib_istream_t::advance(const std::streamsize num_bytes, buffer_t& buffer)
{
        // zlib decompression buffers
        static const std::streamsize chunk_size = 64 * 1024;
        static const auto out_chunk_size = 3 * chunk_size;
        static unsigned char in[chunk_size];
        static unsigned char out[out_chunk_size];

        // decompress the buffer chunk
        while ( static_cast<std::streamsize>(buffer.size()) < num_bytes &&
                m_max_num_bytes > 0 && m_istream)
        {
                const auto to_read = std::min(chunk_size, m_max_num_bytes);
                m_max_num_bytes -= to_read;

                if (!m_istream.read(reinterpret_cast<char*>(in), static_cast<std::streamsize>(to_read)))
                {
                        inflateEnd(&m_zstream);
                        return io_status::error;
                }

                m_zstream.avail_in = static_cast<uInt>(m_istream.gcount());
                m_zstream.next_in = in;

                do
                {
                        m_zstream.avail_out = static_cast<uInt>(out_chunk_size);
                        m_zstream.next_out = out;

                        switch (inflate(&m_zstream, Z_NO_FLUSH))
                        {
                        case Z_STREAM_END:      break;
                        case Z_OK:              break;
                        default:                return io_status::error;
                        }

                        const auto have = out_chunk_size - m_zstream.avail_out;
                        buffer.insert(buffer.end(), out, out + have);
                }
                while (m_zstream.avail_out == 0);
        }

        // OK
        return (m_max_num_bytes == 0) ? io_status::eof : io_status::good;
}
