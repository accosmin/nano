#pragma once

#include "arch.h"
#include <ios>
#include <string>
#include <vector>
#include <functional>

namespace cortex
{
        using buffer_t = std::vector<char>;

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
        /// \brief load the given number of bytes from a stream
        ///
        template
        <
                typename tstream,
                typename tsize
        >
        bool load_buffer_from_stream(tstream& istream, tsize orig_num_bytes, buffer_t& buffer)
        {
                buffer.clear();

                static const std::streamsize chunk_size = 64 * 1024;
                char chunk[chunk_size];

                // read in  chunks
                std::streamsize num_bytes = static_cast<std::streamsize>(orig_num_bytes);
                while (num_bytes > 0 && istream)
                {
                        const std::streamsize to_read = (num_bytes >= chunk_size) ? chunk_size : num_bytes;
                        num_bytes -= to_read;

                        istream.read(chunk, to_read);
                        if (istream.good() || istream.eof())
                        {
                                buffer.insert(buffer.end(), chunk, chunk + istream.gcount());
                        }
                }

                // OK
                return (orig_num_bytes == max_streamsize()) ? istream.eof() : (num_bytes == 0);
        }

        ///
        /// \brief load from a stream until EOF
        ///
        template
        <
                typename tstream
        >
        bool load_buffer_from_stream(tstream& istream, buffer_t& buffer)
        {
                return load_buffer_from_stream(istream, max_streamsize(), buffer);
        }

        ///
        /// \brief save buffer to file
        ///
        NANOCV_PUBLIC bool save_buffer(const std::string& path, const buffer_t& buffer);

        ///
        /// \brief save buffer to file
        ///
        NANOCV_PUBLIC bool load_buffer(const std::string& path, buffer_t& buffer);

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
