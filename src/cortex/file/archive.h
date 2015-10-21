#pragma once

#include "arch.h"
#include "buffer.h"

namespace cortex
{
        ///
        /// \brief decode an archive file (.tar, .gz, .tar.gz, .tar.bz2 etc.)
        ///
        NANOCV_PUBLIC bool unarchive(const std::string& path, const std::string& log_header,
                const buffer_callback_t& callback);

        ///
        /// \brief decode an already opened file file (.tar, .gz, .tar.gz, .tar.bz2 etc.)
        ///
        NANOCV_PUBLIC bool unarchive(std::istream& stream, std::streamsize num_bytes, const std::string& log_header,
                const buffer_callback_t& callback);
}
