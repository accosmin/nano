#pragma once

#include "arch.h"
#include "stringi.h"

namespace nano
{
        ///
        /// \brief returns the lower case string
        ///
        NANO_PUBLIC string_t lower(const string_t& str);

        ///
        /// \brief returns the upper case string
        ///
        NANO_PUBLIC string_t upper(const string_t& str);

        ///
        /// \brief replace all occurencies of a character with another one
        ///
        NANO_PUBLIC string_t replace(const string_t& str, const char token, const char newtoken);

        ///
        /// \brief replace all occurencies of a string with another one
        ///
        NANO_PUBLIC string_t replace(const string_t& str, const string_t& token, const string_t& newtoken);

        ///
        /// \brief check if a string contains a given character
        ///
        NANO_PUBLIC bool contains(const string_t& str, const char token);

        ///
        /// \brief check if two strings are equal (case sensitive)
        ///
        NANO_PUBLIC bool equals(const string_t& str1, const string_t& str2);

        ///
        /// \brief check if two strings are equal (case insensitive)
        ///
        NANO_PUBLIC bool iequals(const string_t& str1, const string_t& str2);

        ///
        /// \brief check if a string starts with a token (case sensitive)
        ///
        NANO_PUBLIC bool starts_with(const string_t& str, const string_t& token);

        ///
        /// \brief check if a string starts with a token (case insensitive)
        ///
        NANO_PUBLIC bool istarts_with(const string_t& str, const string_t& token);

        ///
        /// \brief check if a string ends with a token (case sensitive)
        ///
        NANO_PUBLIC bool ends_with(const string_t& str, const string_t& token);

        ///
        /// \brief check if a string ends with a token (case insensitive)
        ///
        NANO_PUBLIC bool iends_with(const string_t& str, const string_t& token);

        ///
        /// \brief tokenize a string using the given delimeters
        ///
        NANO_PUBLIC strings_t split(const string_t& str, const char* delimeters);

        ///
        /// \brief text alignment options
        ///
        enum class alignment : int
        {
                left,
                center,
                right
        };

        ///
        /// \brief align a string to fill the given size (if possible)
        ///
        NANO_PUBLIC string_t align(const string_t& str, const std::size_t str_size,
                const alignment mode = alignment::left, const char fill_char = ' ');
}
