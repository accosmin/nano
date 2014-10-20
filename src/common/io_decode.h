#pragma once

#include <string>
#include <vector>
#include <functional>

namespace ncv
{
        namespace io
        {
                ///
                /// \brief callback to execute when a file was decompressed from an archive
                ///     - (filename, file content loaded in memory)
                ///
                typedef std::function<void(const std::string&, const std::vector<char>&)>
                        decode_callback_t;

                ///
                /// \brief decode an archive file (.tar, .gz, .tar.gz, .tar.bz2 etc.)
                ///
                bool decode(const std::string& path, const decode_callback_t& callback,
                            const std::string& info_header, const std::string& error_header);
        }
}
