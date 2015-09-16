#pragma once

#include <cctype>
#include <string>
#include <algorithm>

namespace ncv
{
        namespace text
        {
                ///
                /// \brief check if a string ends with a token (case sensitive)
                ///
                inline bool ends_with(const std::string& str, const std::string& token)
                {
                        return  str.size() >= token.size() &&
                                std::equal(token.rbegin(), token.rend(), str.rbegin(),
                                           [] (char c1, char c2) { return c1 == c2; });
                }

                ///
                /// \brief check if a string ends with a token (case insensitive)
                ///
                inline bool iends_with(const std::string& str, const std::string& token)
                {
                        return  str.size() >= token.size() &&
                                std::equal(token.rbegin(), token.rend(), str.rbegin(),
                                           [] (char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
                }
        }
}

