#pragma once

#include "buffer.h"
#include <functional>

struct archive;

namespace nano
{
        ///
        /// \brief wrapper over libarchive to stream binary data.
        ///
        class archive_stream_t
        {
        public:

                archive_stream_t(archive* ar);

                archive_stream_t(const archive_stream_t&) = delete;
                archive_stream_t& operator=(const archive_stream_t&) = delete;

                bool read(char* data, const size_t num_bytes);

        private:

                // attributes
                archive*        m_archive;      ///< libarchive specific
                buffer_t        m_buffer;       ///< buffer
                size_t          m_index;        ///< index in the buffer
        };

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
