#pragma once

#include <iosfwd>
#include <vector>

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using zlib)
                ///
                bool uncompress_zlib(std::istream& istream, std::size_t bytes, std::vector<unsigned char>& data);
        }
}
