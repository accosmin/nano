#pragma once

#include <string>
#include <algorithm>

namespace text
{
        ///
        /// \brief replace a character with another one
        ///
        std::string replace(const std::string& str, const char token, const char newtoken)
        {
                std::string ret = str;
                std::transform(str.begin(), str.end(), ret.begin(),
                               [=] (char c) { return (c == token) ? newtoken : c; });
                return ret;
        }
}

