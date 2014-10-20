#pragma once

#include <string>
#include <vector>
#include <functional>

namespace ncv
{
        class logger_t;

        namespace io
        {
                ///
                /// \brief callback to execute when a file was decompressed from a tar archive
                ///     - (filename, file content loaded in memory)
                ///
                typedef std::function<void(const std::string&, const std::vector<unsigned char>&)>
                        untar_callback_t;

                ///
                /// \brief uncompress a tar archive
                ///
                bool untar(const std::string& path, const untar_callback_t& callback,
                           logger_t& logger_info, logger_t& logger_error);
        }
}
