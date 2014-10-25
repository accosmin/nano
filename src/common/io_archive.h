#pragma once

#include "io_base.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief decode an archive file (.tar, .gz, .tar.gz, .tar.bz2 etc.)
                ///
                bool decode(const std::string& path, const std::string& log_header, const data_callback_t& callback);
        }
}
