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
}
