#ifndef NANOCV_ZCOMPRESS_H
#define NANOCV_ZCOMPRESS_H

#include <iosfwd>
#include <vector>

namespace ncv
{
        ///
        /// \brief uncompress a stream of bytes (using zlib)
        ///
        bool zuncompress(std::istream& istream, std::size_t bytes, std::vector<unsigned char>& data);
}

#endif // NANOCV_ZCOMPRESS_H

