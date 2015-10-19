#pragma once

#include <string>
#include <vector>
#include <functional>

namespace cortex
{
        using buffer_t = std::vector<char>;

        ///
        /// \brief callback to execute when a file was decompressed from an archive
        ///     - (filename, uncompressed file content loaded in memory)
        ///
        using buffer_callback_t = std::function<bool(const std::string&, const buffer_t&)>;
}
