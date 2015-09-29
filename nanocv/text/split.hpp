#pragma once

#include <vector>
#include <string>

namespace text
{
        ///
        /// \brief tokenize a string using the given delimeters
        ///
        inline std::vector<std::string> split(const std::string& str, const char* delimeters)
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

