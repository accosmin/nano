#pragma once

#include "arch.h"
#include "buffer.h"

namespace ncv
{
        ///
        /// \brief uncompress a stream of bytes (using zlib)
        ///
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& istream, std::size_t num_bytes, buffer_t& data);
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& istream, buffer_t& data);
        NANOCV_PUBLIC bool uncompress_gzip(const buffer_t& istream, buffer_t& data);
}
