#pragma once

#include "io_base.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using bzip2)
                ///
                bool uncompress_bzip2(std::istream& istream, size_t num_bytes, data_t& data);
                bool uncompress_bzip2(std::istream& istream, data_t& data);
                bool uncompress_bzip2(const data_t& istream, data_t& data);
        }
}
