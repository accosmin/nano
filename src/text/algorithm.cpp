#include "algorithm.h"
#include <cctype>
#include <algorithm>

namespace zob
{
        std::string lower(const std::string& str)
        {
                std::string ret = str;
                std::transform(str.begin(), str.end(), ret.begin(), [] (char c) { return std::tolower(c); });
                return ret;
        }

        std::string upper(const std::string& str)
        {
                std::string ret = str;
                std::transform(str.begin(), str.end(), ret.begin(), [] (char c) { return std::toupper(c); });
                return ret;
        }

        std::string replace(const std::string& str, const char token, const char newtoken)
        {
                std::string ret = str;
                std::transform(str.begin(), str.end(), ret.begin(),
                               [=] (char c) { return (c == token) ? newtoken : c; });
                return ret;
        }

        bool contains(const std::string& str, const char token)
        {
                return std::find(str.begin(), str.end(), token) != str.end();
        }

        bool equals(const std::string& str1, const std::string& str2)
        {
                return  str1.size() == str2.size() &&
                        std::equal(str1.begin(), str1.end(), str2.begin(),
                                   [] (char c1, char c2) { return c1 == c2; });
        }

        bool iequals(const std::string& str1, const std::string& str2)
        {
                return  str1.size() == str2.size() &&
                        std::equal(str1.begin(), str1.end(), str2.begin(),
                                   [] (char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
        }

        bool starts_with(const std::string& str, const std::string& token)
        {
                return  str.size() >= token.size() &&
                        std::equal(token.begin(), token.end(), str.begin(),
                                   [] (char c1, char c2) { return c1 == c2; });
        }

        bool istarts_with(const std::string& str, const std::string& token)
        {
                return  str.size() >= token.size() &&
                        std::equal(token.begin(), token.end(), str.begin(),
                                   [] (char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
        }

        bool ends_with(const std::string& str, const std::string& token)
        {
                return  str.size() >= token.size() &&
                        std::equal(token.rbegin(), token.rend(), str.rbegin(),
                                   [] (char c1, char c2) { return c1 == c2; });
        }

        bool iends_with(const std::string& str, const std::string& token)
        {
                return  str.size() >= token.size() &&
                        std::equal(token.rbegin(), token.rend(), str.rbegin(),
                                   [] (char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
        }

        std::vector<std::string> split(const std::string& str, const char* delimeters)
        {
                std::vector<std::string> tokens;

                // find the beginning of the splitted strings ...
                auto pos_beg = str.find_first_not_of(delimeters);
                while (pos_beg != std::string::npos)
                {
                        // find the end of the splitted strings ...
                        auto pos_end = str.find_first_of(delimeters, pos_beg + 1);
                        if (pos_end == std::string::npos)
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
}

