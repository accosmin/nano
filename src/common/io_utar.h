#pragma once

#include <string>
#include <vector>
#include <functional>

namespace ncv
{
        namespace io
        {
                ///
                /// \brief callback to execute when a file was decompressed from a tar archive
                ///     - (filename, file content loaded in memory)
                ///
                typedef std::function<void(const std::string&, const std::vector<char>&)>
                        untar_callback_t;

                ///
                /// \brief uncompress a tar archive
                ///
                bool untar(const std::string& path, const untar_callback_t& callback,
                           const std::string& info_header, const std::string& error_header);
        }
}
