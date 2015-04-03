#pragma once

#include "base.h"
#include "../util/arch.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief decode an archive file (.tar, .gz, .tar.gz, .tar.bz2 etc.)
                ///
                NANOCV_DLL_PUBLIC bool decode(const std::string& path, const std::string& log_header,
                        const data_callback_t& callback);
        }
}
