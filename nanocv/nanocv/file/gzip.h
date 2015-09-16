#pragma once

#include "buffer.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using zlib)
                ///
                bool uncompress_gzip(std::istream& istream, size_t num_bytes, buffer_t& data);
                bool uncompress_gzip(std::istream& istream, buffer_t& data);
                bool uncompress_gzip(const buffer_t& istream, buffer_t& data);
        }
}
