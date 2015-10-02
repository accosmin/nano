#pragma once

#include "arch.h"
#include "buffer.h"

namespace ncv
{
        ///
        /// \brief uncompress a stream of bytes (using bzip2)
        ///
        NANOCV_PUBLIC bool uncompress_bzip2(std::istream& istream, std::size_t num_bytes, buffer_t& data);
        NANOCV_PUBLIC bool uncompress_bzip2(std::istream& istream, buffer_t& data);
        NANOCV_PUBLIC bool uncompress_bzip2(const buffer_t& istream, buffer_t& data);
}
