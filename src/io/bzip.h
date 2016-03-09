#pragma once

#include "buffer.h"

namespace zob
{
        class imstream_t;

        ///
        /// \brief uncompress a stream of bytes (using bzip2)
        ///
        ZOB_PUBLIC bool uncompress_bzip2(std::istream& istream, std::streamsize num_bytes, buffer_t&);
        ZOB_PUBLIC bool uncompress_bzip2(std::istream& istream, buffer_t&);
        ZOB_PUBLIC bool uncompress_bzip2(imstream_t& istream, buffer_t&);
}
