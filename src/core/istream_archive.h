#pragma once

#include "istream.h"

struct archive;

namespace nano
{
        ///
        /// \brief libarchive-based streaming of binary data.
        ///
        struct NANO_PUBLIC archive_istream_t final : public istream_t
        {
                explicit archive_istream_t(archive* ar);

                ~archive_istream_t() override = default;

                io_status advance(const std::streamsize num_bytes, buffer_t& buffer) override;

        private:

                // attributes
                archive*                m_archive;      ///< libarchive specific
                std::vector<char>       m_buffer;       ///< buffer
        };
}
