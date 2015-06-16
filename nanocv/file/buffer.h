#pragma once

#include <string>
#include <vector>
#include <iosfwd>
#include <cstring>
#include <functional>

namespace ncv
{
        namespace io
        {
                using std::size_t;
                typedef std::vector<char> buffer_t;

                ///
                /// \brief callback to execute when a file was decompressed from an archive
                ///     - (filename, uncompressed file content loaded in memory)
                ///
                typedef std::function<bool(const std::string&, const buffer_t&)> buffer_callback_t;
        }
}
