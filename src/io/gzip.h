#pragma once

#include "buffer.h"

namespace zob
{
        class imstream_t;

        ///
        /// \brief uncompress a stream of bytes (using zlib)
        ///
        ZOB_PUBLIC bool uncompress_gzip(std::istream& istream, std::streamsize num_bytes, buffer_t&);
        ZOB_PUBLIC bool uncompress_gzip(std::istream& istream, buffer_t&);
        ZOB_PUBLIC bool uncompress_gzip(imstream_t& istream, buffer_t&);
}
