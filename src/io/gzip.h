#pragma once

#include "buffer.h"

namespace io
{
        class imstream_t;

        ///
        /// \brief uncompress a stream of bytes (using zlib)
        ///
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& istream, std::streamsize num_bytes, buffer_t&);
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& istream, buffer_t&);
        NANOCV_PUBLIC bool uncompress_gzip(imstream_t& istream, buffer_t&);
}
