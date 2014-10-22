#pragma once

#include "io_base.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using zlib)
                ///
                bool uncompress_gzip(std::istream& istream, size_t num_bytes, data_t& data);
                bool uncompress_gzip(std::istream& istream, data_t& data);
                bool uncompress_gzip(const std::string& path, data_t& data);

                bool uncompress_gzip(const data_t& istream, data_t& data);
        }
}
