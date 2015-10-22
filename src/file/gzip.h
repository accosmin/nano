#pragma once

#include "buffer.h"

namespace file
{
        class mstream_t;

        ///
        /// \brief uncompress a stream of bytes (using zlib)
        ///
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& istream, std::streamsize num_bytes, buffer_t&);
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& istream, buffer_t&);
        NANOCV_PUBLIC bool uncompress_gzip(mstream_t& istream, buffer_t&);
}
