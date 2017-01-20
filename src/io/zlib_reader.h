#pragma once

#include "buffer.h"
#include <zlib.h>

namespace nano
{
        ///
        /// \brief zlib-based streaming of gzip-compressed binary data.
        ///
        class NANO_PUBLIC zlib_reader_t
        {
        public:

                zlib_reader_t(std::istream& istream, const std::streamsize max_num_bytes = max_streamsize());

                zlib_reader_t(const zlib_reader_t&) = delete;
                zlib_reader_t& operator=(const zlib_reader_t&) = delete;

                ~zlib_reader_t();

                io_status advance(const std::streamsize num_bytes, buffer_t& buffer);

        private:

                // attributes
                std::istream&   m_istream;              ///< input stream
                std::streamsize m_max_num_bytes;        ///< maximum number of bytes to read from the input stream
                z_stream        m_zstream;              ///< zlib stream
        };
}
