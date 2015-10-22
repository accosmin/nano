#pragma once

#include "buffer.h"
#include "mstream.h"

namespace cortex
{
        ///
        /// \brief uncompress a stream of bytes (using zlib)
        ///
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& istream, std::streamsize num_bytes, buffer_t& data);
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& istream, buffer_t& data);
        NANOCV_PUBLIC bool uncompress_gzip(mstream_t& istream, buffer_t& data);
}
