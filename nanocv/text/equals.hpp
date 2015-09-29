#pragma once

#include <cctype>
#include <string>
#include <algorithm>

namespace text
{
        ///
        /// \brief check if two strings are equal (case sensitive)
        ///
        inline bool equals(const std::string& str1, const std::string& str2)
        {
                return  str1.size() == str2.size() &&
                        std::equal(str1.begin(), str1.end(), str2.begin(),
                                   [] (char c1, char c2) { return c1 == c2; });
        }

        ///
        /// \brief check if two strings are equal (case insensitive)
        ///
        inline bool iequals(const std::string& str1, const std::string& str2)
        {
                return  str1.size() == str2.size() &&
                        std::equal(str1.begin(), str1.end(), str2.begin(),
                                   [] (char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
        }
}

