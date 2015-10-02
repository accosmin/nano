#pragma once

#include <cctype>
#include <string>
#include <algorithm>

namespace text
{
        ///
        /// \brief returns the upper case string
        ///
        inline std::string upper(const std::string& str)
        {
                std::string ret = str;
                std::transform(str.begin(), str.end(), ret.begin(), [] (char c) { return std::toupper(c); });
                return ret;
        }
}

