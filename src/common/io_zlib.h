#pragma once

#include "io.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using zlib)
                ///
                bool uncompress_zlib(std::istream& istream, std::size_t bytes, data_t& data);
                bool uncompress_zlib(std::istream& istream, data_t& data);
                bool uncompress_zlib(const std::string& path, data_t& data);
        }
}
