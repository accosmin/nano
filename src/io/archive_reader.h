#pragma once

#include "buffer.h"

struct archive;

namespace nano
{
        ///
        /// \brief libarchive-based streaming of binary data.
        ///
        class NANO_PUBLIC archive_reader_t
        {
        public:

                archive_reader_t(archive* ar);

                archive_reader_t(const archive_reader_t&) = delete;
                archive_reader_t& operator=(const archive_reader_t&) = delete;

                ~archive_reader_t() = default;

                io_status advance(const std::streamsize num_bytes, buffer_t& buffer);

        private:

                // attributes
                archive*                m_archive;      ///< libarchive specific
                std::vector<char>       m_buffer;       ///< buffer
                std::streamsize         m_index;        ///< index in the buffer
        };
}
