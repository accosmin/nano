#pragma once

#include <string>

namespace nano
{
        /// todo: remove this file and use <filesystem> with C++17.

        ///
        /// \brief extracts file name from path (e.g. /usr/include/file.ext -> file.ext).
        ///
        inline std::string filename(const std::string& path)
        {
                const auto pos = path.find_last_of("/\\");
                return (pos == std::string::npos) ? path : path.substr(pos + 1);
        }

        ///
        /// \brief extracts file extension from path (e.g. /usr/include/file.ext -> ext).
        ///
        inline std::string extension(const std::string& path)
        {
                const auto pos = path.find_last_of('.');
                return (pos == std::string::npos) ? std::string() : path.substr(pos + 1);
        }

        ///
        /// \brief extracts directory name from path (e.g. /usr/include/file.ext -> /usr/include/).
        ///
        inline std::string dirname(const std::string& path)
        {
                const auto pos = path.find_last_of("/\\");
                return (pos == std::string::npos) ? "./" : path.substr(0, pos + 1);
        }

        ///
        /// \brief extracts file stem from path (e.g. /usr/include/file.ext -> file).
        ///
        inline std::string stem(const std::string& path)
        {
                const auto pos_dir = path.find_last_of("/\\");
                const auto pos_ext = path.find_last_of('.');

                if (pos_dir == std::string::npos)
                {
                        return  (pos_ext == std::string::npos) ?
                                path : path.substr(0, pos_ext);
                }
                else
                {
                        return  (pos_ext == std::string::npos) ?
                                path.substr(pos_dir + 1) : path.substr(pos_dir + 1, pos_ext - pos_dir - 1);
                }
        }
}
