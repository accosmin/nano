#pragma once

#include <ios>
#include "arch.h"
#include <vector>
#include <string>

namespace nano
{
        using buffer_t = std::vector<char>;
        using string_t = std::string;

        enum class io_status
        {
                good,
                eof,
                error
        };

        ///
        /// \brief allocates a buffer of the given size.
        ///
        template <typename tsize>
        buffer_t make_buffer(const tsize size)
        {
                return buffer_t(static_cast<std::size_t>(size));
        }

        ///
        /// \brief maximum file/stream size in bytes (useful for indicating a read-until-EOF condition).
        ///
        NANO_PUBLIC std::streamsize max_streamsize();

        ///
        /// \brief save binary buffer or string to file.
        ///
        NANO_PUBLIC bool save_buffer(const string_t& path, const buffer_t&);
        NANO_PUBLIC bool save_string(const string_t& path, const string_t&);

        ///
        /// \brief load binary buffer or string from file.
        ///
        NANO_PUBLIC bool load_buffer(const string_t& path, buffer_t&);
        NANO_PUBLIC bool load_string(const string_t& path, string_t&);
}
