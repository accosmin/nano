#pragma once

#include <iosfwd>
#include <vector>

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using bzip2)
                ///
                bool uncompress_bzip2(std::istream& istream, std::size_t bytes, std::vector<char>& data);
        }
}
