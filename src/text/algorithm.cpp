#include "algorithm.h"
#include <cctype>
#include <algorithm>

using namespace nano;

static inline bool icequals(const char c1, const char c2)
{
        return std::tolower(c1) == std::tolower(c2);
}

string_t nano::lower(const string_t& str)
{
        string_t ret = str;
        std::transform(str.begin(), str.end(), ret.begin(), [] (const char c) { return std::tolower(c); });
        return ret;
}

string_t nano::upper(const string_t& str)
{
        string_t ret = str;
        std::transform(str.begin(), str.end(), ret.begin(), [] (const char c) { return std::toupper(c); });
        return ret;
}

string_t nano::replace(const string_t& str, const char token, const char newtoken)
{
        string_t ret = str;
        std::transform(str.begin(), str.end(), ret.begin(),
                       [=] (const char c) { return (c == token) ? newtoken : c; });
        return ret;
}

string_t nano::replace(const string_t& str, const string_t& token, const string_t& newtoken)
{
        string_t ret = str;
        for (size_t index = 0;;)
        {
                index = ret.find(token, index);
                if (index == string_t::npos)
                {
                        break;
                }
                ret.replace(index, token.size(), newtoken);
                index += newtoken.size();
        }
        return ret;
}

bool nano::contains(const string_t& str, const char token)
{
        return std::find(str.begin(), str.end(), token) != str.end();
}

bool nano::equals(const string_t& str1, const string_t& str2)
{
        return  str1.size() == str2.size() &&
                std::equal(str1.begin(), str1.end(), str2.begin());
}

bool nano::iequals(const string_t& str1, const string_t& str2)
{
        return  str1.size() == str2.size() &&
                std::equal(str1.begin(), str1.end(), str2.begin(), icequals);
}

bool nano::starts_with(const string_t& str, const string_t& token)
{
        return  str.size() >= token.size() &&
                std::equal(token.begin(), token.end(), str.begin());
}

bool nano::istarts_with(const string_t& str, const string_t& token)
{
        return  str.size() >= token.size() &&
                std::equal(token.begin(), token.end(), str.begin(), icequals);
}

bool nano::ends_with(const string_t& str, const string_t& token)
{
        return  str.size() >= token.size() &&
                std::equal(token.rbegin(), token.rend(), str.rbegin());
}

bool nano::iends_with(const string_t& str, const string_t& token)
{
        return  str.size() >= token.size() &&
                std::equal(token.rbegin(), token.rend(), str.rbegin(), icequals);
}

strings_t nano::split(const string_t& str, const char* delimeters)
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

string_t nano::align(const string_t& str, const std::size_t str_size, const alignment mode, const char fill_char)
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
