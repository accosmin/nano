#pragma once

#include "istream.h"
#include "archive_reader.h"
#include <functional>

namespace nano
{
        ///
        /// \brief wrapper over libarchive to stream binary data.
        ///
        using archive_stream_t = istream_t<archive_reader_t>;

        ///
        /// \brief callback to execute when a file was decompressed from an archive
        ///     - (filename, binary streaming)
        ///     - returns true if it should continue
        ///
        using archive_callback_t = std::function<bool(const std::string&, archive_stream_t&)>;

        ///
        /// \brief callback to execute when an error was detected at decompression
        ///     - (error message)
        ///
        using archive_error_callback_t = std::function<void(const std::string&)>;

        ///
        /// \brief decode an archive file (.tar, .gz, .tar.gz, .tar.bz2 etc.)
        ///
        NANO_PUBLIC bool unarchive(const std::string& path,
                const archive_callback_t&, const archive_error_callback_t&);
}
