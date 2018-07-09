#pragma once

#include "stringi.h"    /// todo: replace this include with <vector> & <string>
#include <cctype>
#include <algorithm>

namespace nano
{
        /// todo: replace these filesystem utilities with <filesystem> in C++17.

        ///
        /// \brief extracts file name from path (e.g. /usr/include/file.ext -> file.ext).
        ///
        inline string_t filename(const string_t& path)
        {
                const auto pos = path.find_last_of("/\\");
                return (pos == string_t::npos) ? path : path.substr(pos + 1);
        }

        ///
        /// \brief extracts file extension from path (e.g. /usr/include/file.ext -> ext).
        ///
        inline string_t extension(const string_t& path)
        {
                const auto pos = path.find_last_of('.');
                return (pos == string_t::npos) ? string_t() : path.substr(pos + 1);
        }

        ///
        /// \brief extracts directory name from path (e.g. /usr/include/file.ext -> /usr/include/).
        ///
        inline string_t dirname(const string_t& path)
        {
                const auto pos = path.find_last_of("/\\");
                return (pos == string_t::npos) ? "./" : path.substr(0, pos + 1);
        }

        ///
        /// \brief extracts file stem from path (e.g. /usr/include/file.ext -> file).
        ///
        inline string_t stem(const string_t& path)
        {
                const auto pos_dir = path.find_last_of("/\\");
                const auto pos_ext = path.find_last_of('.');

                if (pos_dir == string_t::npos)
                {
                        return  (pos_ext == string_t::npos) ?
                                path : path.substr(0, pos_ext);
                }
                else
                {
                        return  (pos_ext == string_t::npos) ?
                                path.substr(pos_dir + 1) : path.substr(pos_dir + 1, pos_ext - pos_dir - 1);
                }
        }

        ///
        /// \brief returns the lower case string
        ///
        inline string_t lower(string_t str)
        {
                std::transform(str.begin(), str.end(), str.begin(),
                               [] (const unsigned char c) { return std::tolower(c); });
                return str;
        }

        ///
        /// \brief returns the upper case string
        ///
        inline string_t upper(string_t str)
        {
                std::transform(str.begin(), str.end(), str.begin(),
                               [] (const unsigned char c) { return std::toupper(c); });
                return str;
        }

        ///
        /// \brief replace all occurencies of a character with another one
        ///
        inline string_t replace(string_t str, const char token, const char newtoken)
        {
                std::transform(str.begin(), str.end(), str.begin(),
                               [=] (const char c) { return (c == token) ? newtoken : c; });
                return str;
        }

        ///
        /// \brief replace all occurencies of a string with another one
        ///
        inline string_t replace(string_t str, const string_t& token, const string_t& newtoken)
        {
                for (size_t index = 0;;)
                {
                        index = str.find(token, index);
                        if (index == string_t::npos)
                        {
                                break;
                        }
                        str.replace(index, token.size(), newtoken);
                        index += newtoken.size();
                }
                return str;
        }

        ///
        /// \brief check if two characters are equal case-insensitively
        ///
        inline bool iequal(const unsigned char c1, const unsigned char c2)
        {
                return std::tolower(c1) == std::tolower(c2);
        }

        ///
        /// \brief check if a string contains a given character
        ///
        inline bool contains(const string_t& str, const char token)
        {
                return std::find(str.begin(), str.end(), token) != str.end();
        }

        ///
        /// \brief check if two strings are equal (case sensitive)
        ///
        inline bool equals(const string_t& str1, const string_t& str2)
        {
                return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin());
        }

        ///
        /// \brief check if two strings are equal (case insensitive)
        ///
        inline bool iequals(const string_t& str1, const string_t& str2)
        {
                return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin(), iequal);
        }

        ///
        /// \brief check if a string starts with a token (case sensitive)
        ///
        inline bool starts_with(const string_t& str, const string_t& token)
        {
                return str.size() >= token.size() && std::equal(token.begin(), token.end(), str.begin());
        }

        ///
        /// \brief check if a string starts with a token (case insensitive)
        ///
        inline bool istarts_with(const string_t& str, const string_t& token)
        {
                return str.size() >= token.size() && std::equal(token.begin(), token.end(), str.begin(), iequal);
        }

        ///
        /// \brief check if a string ends with a token (case sensitive)
        ///
        inline bool ends_with(const string_t& str, const string_t& token)
        {
                return str.size() >= token.size() && std::equal(token.rbegin(), token.rend(), str.rbegin());
        }

        ///
        /// \brief check if a string ends with a token (case insensitive)
        ///
        inline bool iends_with(const string_t& str, const string_t& token)
        {
                return str.size() >= token.size() && std::equal(token.rbegin(), token.rend(), str.rbegin(), iequal);
        }

        ///
        /// \brief tokenize a string using the given delimeters
        ///
        inline strings_t split(const string_t& str, const char* delimeters)
        {
                strings_t tokens;

                // find the beginning of the splitted strings ...
                auto pos_beg = str.find_first_not_of(delimeters);
                while (pos_beg != string_t::npos)
                {
                        // find the end of the splitted strings ...
                        auto pos_end = str.find_first_of(delimeters, pos_beg + 1);
                        if (pos_end == string_t::npos)
                                pos_end = str.size();
                        if (pos_end != pos_beg)
                                tokens.emplace_back(str.substr(pos_beg, pos_end - pos_beg));

                        // continue to iterate for the next splitted string
                        pos_beg = str.find_first_not_of(delimeters, pos_end);
                }

                if (tokens.empty())
                {
                        tokens.push_back(str);
                }

                return tokens;
        }

        inline strings_t split(const string_t& str, const char delimeter)
        {
                const char delimeters[2] = {delimeter, '\0'};
                return split(str, delimeters);
        }

        ///
        /// \brief align a string to fill the given size (if possible)
        ///
        inline string_t align(const string_t& str, const size_t str_size,
               const alignment mode = alignment::left, const char fill_char = ' ')
        {
                const auto fill_size = (str.size() > str_size) ? (0) : (str_size - str.size());

                switch (mode)
                {
                case alignment::center:
                        return string_t(fill_size / 2, fill_char) +
                               str +
                               string_t(fill_size - fill_size / 2, fill_char);

                case alignment::right:
                        return string_t(fill_size, fill_char) +
                               str;

                case alignment::left:
                default:
                        return str +
                               string_t(fill_size, fill_char);
                }
        }
}
