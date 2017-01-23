#pragma once

#include <ios>
#include "arch.h"
#include <limits>
#include <vector>
#include <string>

namespace nano
{
        using buffer_t = std::vector<char>;

        enum class io_status
        {
                good,
                eof,
                error
        };

        ///
        /// \brief allocates a buffer of the given size
        ///
        template <typename tsize>
        buffer_t make_buffer(const tsize size)
        {
                return buffer_t(static_cast<std::size_t>(size));
        }

        ///
        /// \brief maximum file/stream size in bytes (useful for indicating a read-until-EOF condition)
        ///
        inline std::streamsize max_streamsize()
        {
                return std::numeric_limits<std::streamsize>::max();
        }

        ///
        /// \brief save buffer to file
        ///
        NANO_PUBLIC bool save_buffer(const std::string& path, const buffer_t& buffer);

        ///
        /// \brief load buffer from file
        ///
        NANO_PUBLIC bool load_buffer(const std::string& path, buffer_t& buffer);
}
