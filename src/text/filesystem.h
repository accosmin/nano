#pragma once

#include "arch.h"
#include <string>

namespace text
{
        ///
        /// \brief extracts file name from path (e.g. /usr/include/file.ext -> file.ext).
        ///
        ZOB_PUBLIC std::string filename(const std::string& path);

        ///
        /// \brief extracts file extension from path (e.g. /usr/include/file.ext -> ext).
        ///
        ZOB_PUBLIC std::string extension(const std::string& path);

        ///
        /// \brief extracts file stem from path (e.g. /usr/include/file.ext -> file).
        ///
        ZOB_PUBLIC std::string stem(const std::string& path);

        ///
        /// \brief extracts directory name from path (e.g. /usr/include/file.ext -> /usr/include/).
        ///
        ZOB_PUBLIC std::string dirname(const std::string& path);
}

