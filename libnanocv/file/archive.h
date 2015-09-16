#pragma once

#include "buffer.h"
#include "libcore/arch.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief decode an archive file (.tar, .gz, .tar.gz, .tar.bz2 etc.)
                ///
                NANOCV_PUBLIC bool decode(const std::string& path, const std::string& log_header,
                        const buffer_callback_t& callback);
        }
}
