#pragma once

#include "arch.h"
#include "stringi.h"

namespace nano
{
        ///
        /// \brief extracts file name from path (e.g. /usr/include/file.ext -> file.ext).
        ///
        NANO_PUBLIC string_t filename(const string_t& path);

        ///
        /// \brief extracts file extension from path (e.g. /usr/include/file.ext -> ext).
        ///
        NANO_PUBLIC string_t extension(const string_t& path);

        ///
        /// \brief extracts file stem from path (e.g. /usr/include/file.ext -> file).
        ///
        NANO_PUBLIC string_t stem(const string_t& path);

        ///
        /// \brief extracts directory name from path (e.g. /usr/include/file.ext -> /usr/include/).
        ///
        NANO_PUBLIC string_t dirname(const string_t& path);
}

