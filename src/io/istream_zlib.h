#pragma once

#include "istream.h"
#include <zlib.h>

namespace nano
{
        ///
        /// \brief zlib-based streaming of gzip-compressed binary data.
        ///
        class zlib_istream_t : public istream_t
        {
        public:

                zlib_istream_t(std::istream& istream, const std::streamsize max_num_bytes = max_streamsize()) :
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

                ~zlib_istream_t()
                {
                        inflateEnd(&m_zstream);
                }

        protected:

                virtual status advance(const std::streamsize num_bytes, buffer_t& buffer) override final
                {
                        // zlib decompression buffers
                        static const std::streamsize chunk_size = 64 * 1024;
                        static const auto out_chunk_size = 3 * chunk_size;
                        static unsigned char in[chunk_size];
                        static unsigned char out[out_chunk_size];

                        // decompress the buffer chunk
                        while (m_max_num_bytes > 0 && m_istream && buffer.size() < num_bytes)
                        {
                                const auto to_read = (m_max_num_bytes >= chunk_size) ? chunk_size : m_max_num_bytes;
                                m_max_num_bytes -= to_read;

                                if (!m_istream.read(reinterpret_cast<char*>(in), to_read))
                                {
                                        inflateEnd(&m_zstream);
                                        return status::error;
                                }

                                m_zstream.avail_in = static_cast<uInt>(m_istream.gcount());
                                m_zstream.next_in = in;

                                do
                                {
                                        m_zstream.avail_out = static_cast<uInt>(out_chunk_size);
                                        m_zstream.next_out = out;

                                        const int ret = inflate(&m_zstream, Z_NO_FLUSH);
                                        if (ret != Z_OK && ret != Z_STREAM_END)
                                        {
                                                return status::error;
                                        }

                                        const std::streamsize have = out_chunk_size - m_zstream.avail_out;
                                        buffer.insert(buffer.end(), out, out + have);
                                }
                                while (m_zstream.avail_out == 0);
                        }

                        // todo: EOF not detected properly!

                        // OK
                        return (buffer.size() >= num_bytes) ? status::ok : status::error;
                }

        private:

                // attributes
                std::istream&   m_istream;              ///< input stream
                std::streamsize m_max_num_bytes;        ///< maximum number of bytes to read from the input stream
                z_stream        m_zstream;              ///< zlib stream
        };
}
