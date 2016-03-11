#pragma once

#include "buffer.h"

namespace nano
{
        class imstream_t;

        ///
        /// \brief uncompress a stream of bytes (using zlib)
        ///
        NANO_PUBLIC bool uncompress_gzip(std::istream& istream, std::streamsize num_bytes, buffer_t&);
        NANO_PUBLIC bool uncompress_gzip(std::istream& istream, buffer_t&);
        NANO_PUBLIC bool uncompress_gzip(imstream_t& istream, buffer_t&);
}
