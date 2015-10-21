#pragma once

#include "arch.h"
#include "buffer.h"

namespace cortex
{
        ///
        /// \brief decode an archive file (.tar, .gz, .tar.gz, .tar.bz2 etc.)
        ///
        NANOCV_PUBLIC bool unarchive(const std::string& path,
                const archive_callback_t&, const archive_error_callback_t&);

        ///
        /// \brief uncompress a gzip chunk
        ///
        NANOCV_PUBLIC bool uncompress_gzip(std::istream& stream, std::streamsize num_bytes,
                const archive_callback_t&, const archive_error_callback_t&);
}
