#pragma once

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

        ////
        /// \brief maximum stream size in bytes (useful to indicate that reading should be done until EOF)
        ///
        inline std::streamsize max_streamsize()
        {
                return std::numeric_limits<std::streamsize>::max();
        }

        ///
        /// \brief callback to execute when a file was decompressed from an archive
        ///     - (filename, uncompressed file content loaded in memory)
        ///
        using buffer_callback_t = std::function<bool(const std::string&, const buffer_t&)>;
}
