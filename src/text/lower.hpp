#pragma once

#include <cctype>
#include <string>
#include <algorithm>

namespace text
{
        ///
        /// \brief returns the lower case string
        ///
        std::string lower(const std::string& str)
        {
                std::string ret = str;
                std::transform(str.begin(), str.end(), ret.begin(), [] (char c) { return std::tolower(c); });
                return ret;
        }
}

