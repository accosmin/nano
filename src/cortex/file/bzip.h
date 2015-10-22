#pragma once

#include "buffer.h"
#include "mstream.h"

namespace cortex
{
        ///
        /// \brief uncompress a stream of bytes (using bzip2)
        ///
        NANOCV_PUBLIC bool uncompress_bzip2(std::istream& istream, std::streamsize num_bytes, buffer_t& data);
        NANOCV_PUBLIC bool uncompress_bzip2(std::istream& istream, buffer_t& data);
        NANOCV_PUBLIC bool uncompress_bzip2(mstream_t& istream, buffer_t& data);
}
