#pragma once

#include "arch.h"
#include <ios>
#include <iosfwd>
#include <string>
#include <vector>

namespace file
{
        using buffer_t = std::vector<char>;

        class mstream_t;

        ///
        /// \brief allocates a buffer of the given size
        ///
        template
        <
                typename tsize
        >
        buffer_t make_buffer(const tsize size)
        {
                return buffer_t(static_cast<std::size_t>(size));
        }

        ///
        /// \brief maximum file/stream size in bytes (useful for indicating a read-until-EOF condition)
        ///
        NANOCV_PUBLIC std::streamsize max_streamsize();

        ///
        /// \brief load a stream of bytes
        ///
        NANOCV_PUBLIC bool load_buffer_from_stream(std::istream& istream, std::streamsize num_bytes, buffer_t&);
        NANOCV_PUBLIC bool load_buffer_from_stream(std::istream& istream, buffer_t&);
        NANOCV_PUBLIC bool load_buffer_from_stream(mstream_t& istream, buffer_t&);

        ///
        /// \brief save buffer to file
        ///
        NANOCV_PUBLIC bool save_buffer(const std::string& path, const buffer_t& buffer);

        ///
        /// \brief load buffer from file
        ///
        NANOCV_PUBLIC bool load_buffer(const std::string& path, buffer_t& buffer);

        ///
        /// \brief save string to binary file
        ///
        NANOCV_PUBLIC bool save_string(std::ostream&, const std::string& str);

        ///
        /// \brief load string from binary file
        ///
        NANOCV_PUBLIC std::string load_string(std::istream&);
}
