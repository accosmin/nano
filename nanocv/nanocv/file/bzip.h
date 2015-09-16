#pragma once

#include "buffer.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using bzip2)
                ///
                bool uncompress_bzip2(std::istream& istream, size_t num_bytes, buffer_t& data);
                bool uncompress_bzip2(std::istream& istream, buffer_t& data);
                bool uncompress_bzip2(const buffer_t& istream, buffer_t& data);
        }
}
