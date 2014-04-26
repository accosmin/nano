#ifndef NANOCV_IO_ZLIB
#define NANOCV_IO_ZLIB

#include <iosfwd>
#include <vector>

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using zlib)
                ///
                bool zuncompress(std::istream& istream, std::size_t bytes, std::vector<unsigned char>& data);
        }
}

#endif // NANOCV_IO_ZLIB

