#pragma once

#include "buffer.h"
#include <functional>

namespace nano
{
        ///
        /// \brief callback to execute when a file was decompressed from an archive
        ///     - (filename, uncompressed file content loaded in memory)
        ///     - returns true if it should continue
        ///
        using archive_callback_t = std::function<bool(const std::string&, const buffer_t&)>;

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
