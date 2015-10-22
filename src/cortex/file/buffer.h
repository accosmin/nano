#pragma once

#include "arch.h"
#include <ios>
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
        template
        <
                typename tstream
        >
        bool load_buffer(tstream& stream, std::streamsize num_bytes, buffer_t& buffer)
        {
                buffer.resize(static_cast<std::size_t>(num_bytes));

                const std::streamsize stream_size = 64 * 1024;

                char* pbuffer = buffer.data();
                while (stream && num_bytes > 0)
                {
                        const auto to_read = num_bytes >= stream_size ? stream_size : num_bytes;

                        if (stream.read(pbuffer, to_read))
                        {
                                pbuffer += to_read;
                                num_bytes -= to_read;
                        }
                }

                return num_bytes == 0;
        }

        ///
        /// \brief callback to execute when a file was decompressed from an archive
        ///     - (filename, uncompressed file content loaded in memory)
        ///     - returns true if it should continue
        ///
        using archive_callback_t = std::function<bool(const std::string&, const buffer_t&)>;

        ///
        /// \brief callback to execute when an error was detected at decompressin
        ///     - (error message)
        ///
        using archive_error_callback_t = std::function<void(const std::string&)>;
}
