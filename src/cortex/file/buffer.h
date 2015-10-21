#pragma once

#include "arch.h"
#include <ios>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>
#include <functional>

namespace cortex
{
        using buffer_t = std::vector<char>;

        template
        <
                typename tsize
        >
        buffer_t make_buffer(const tsize size)
        {
                return buffer_t(static_cast<std::size_t>(size));
        }

        ///
        /// \brief load the given number of bytes from a stream
        ///
        NANOCV_PUBLIC bool load_buffer(std::istream& stream, std::streamsize num_bytes, buffer_t&);

        ///
        /// \brief callback to execute when a file was decompressed from an archive
        ///     - (filename, uncompressed file content loaded in memory)
        ///
        using buffer_callback_t = std::function<bool(const std::string&, const buffer_t&)>;
}
