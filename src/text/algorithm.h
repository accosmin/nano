#pragma once

#include "arch.h"
#include <string>
#include <vector>

namespace nano
{
        ///
        /// \brief returns the lower case string
        ///
        NANO_PUBLIC std::string lower(const std::string& str);

        ///
        /// \brief returns the upper case string
        ///
        NANO_PUBLIC std::string upper(const std::string& str);

        ///
        /// \brief replace all occurencies of a character with another one
        ///
        NANO_PUBLIC std::string replace(const std::string& str, const char token, const char newtoken);

        ///
        /// \brief replace all occurencies of a string with another one
        ///
        NANO_PUBLIC std::string replace(const std::string& str, const std::string& token, const std::string& newtoken);

        ///
        /// \brief check if a string contains a given character
        ///
        NANO_PUBLIC bool contains(const std::string& str, const char token);

        ///
        /// \brief check if two strings are equal (case sensitive)
        ///
        NANO_PUBLIC bool equals(const std::string& str1, const std::string& str2);

        ///
        /// \brief check if two strings are equal (case insensitive)
        ///
        NANO_PUBLIC bool iequals(const std::string& str1, const std::string& str2);

        ///
        /// \brief check if a string starts with a token (case sensitive)
        ///
        NANO_PUBLIC bool starts_with(const std::string& str, const std::string& token);

        ///
        /// \brief check if a string starts with a token (case insensitive)
        ///
        NANO_PUBLIC bool istarts_with(const std::string& str, const std::string& token);

        ///
        /// \brief check if a string ends with a token (case sensitive)
        ///
        NANO_PUBLIC bool ends_with(const std::string& str, const std::string& token);

        ///
        /// \brief check if a string ends with a token (case insensitive)
        ///
        NANO_PUBLIC bool iends_with(const std::string& str, const std::string& token);

        ///
        /// \brief tokenize a string using the given delimeters
        ///
        NANO_PUBLIC std::vector<std::string> split(const std::string& str, const char* delimeters);
}

