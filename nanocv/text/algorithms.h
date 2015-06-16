#pragma once

#include <vector>
#include <string>
#include "nanocv/arch.h"

namespace ncv
{
        ///
        /// \brief text alignment options
        ///
        enum class align : int
        {
                left,
                center,
                right
        };

        namespace text
        {
                ///
                /// \brief align a string to fill the given size
                ///
                NANOCV_PUBLIC std::string resize(const std::string& str, std::size_t size,
                        align alignment = align::left, char fill_char = ' ');

                ///
                /// \brief tokenize a string using the given delimeters
                ///
                NANOCV_PUBLIC std::vector<std::string> split(const std::string& str, const char* delimeters);

                ///
                /// \brief returns the lower case string
                ///
                NANOCV_PUBLIC std::string lower(const std::string& str);

                ///
                /// \brief returns the upper case string
                ///
                NANOCV_PUBLIC std::string upper(const std::string& str);

                ///
                /// \brief check if a string ends with a token (case sensitive)
                ///
                NANOCV_PUBLIC bool ends_with(const std::string& str, const std::string& token);

                ///
                /// \brief check if a string ends with a token (case insensitive)
                ///
                NANOCV_PUBLIC bool iends_with(const std::string& str, const std::string& token);

                ///
                /// \brief check if two strings are equal (case sensitive)
                ///
                NANOCV_PUBLIC bool equals(const std::string& str1, const std::string& str2);

                ///
                /// \brief check if two strings are equal (case insensitive)
                ///
                NANOCV_PUBLIC bool iequals(const std::string& str1, const std::string& str2);
        }
}

