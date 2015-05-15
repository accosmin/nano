#pragma once

#include "arch.h"
#include <vector>
#include <string>

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

                ///
                /// \brief cast to string for built-in types
                ///
                template
                <
                        typename tvalue
                >
                std::string to_string(tvalue value)
                {
                        return std::to_string(value);
                }
                template <>
                inline std::string to_string(std::string value)
                {
                        return value;
                }
                template <>
                inline std::string to_string(const char* value)
                {
                        return value;
                }

                ///
                /// \brief cast built-int types from string
                ///
                template
                <
                        typename tvalue
                >
                tvalue from_string(const std::string& str);
                template <>
                inline short from_string<short>(const std::string& str)
                {
                        return std::stoi(str);
                }
                template <>
                inline int from_string<int>(const std::string& str)
                {
                        return std::stoi(str);
                }
                template <>
                inline long from_string<long>(const std::string& str)
                {
                        return std::stol(str);
                }
                template <>
                inline long long from_string<long long>(const std::string& str)
                {
                        return std::stoll(str);
                }
                template <>
                inline unsigned long from_string<unsigned long>(const std::string& str)
                {
                        return std::stoul(str);
                }
                template <>
                inline unsigned long long from_string<unsigned long long>(const std::string& str)
                {
                        return std::stoull(str);
                }
                template <>
                inline float from_string<float>(const std::string& str)
                {
                        return std::stof(str);
                }
                template <>
                inline double from_string<double>(const std::string& str)
                {
                        return std::stod(str);
                }
                template <>
                inline long double from_string<long double>(const std::string& str)
                {
                        return std::stold(str);
                }
                template <>
                inline std::string from_string<std::string>(const std::string& str)
                {
                        return str;
                }

                ///
                /// \brief compact a list of values into a string using the given "glue" string
                ///
                template
                <
                        typename tcontainer,
                        typename tvalue = decltype(*std::begin(tcontainer()))
                >
                std::string concatenate(const tcontainer& values, const std::string& glue = ",")
                {
                        std::string ret;
                        for (auto value : values)
                        {
                                ret += to_string(value) + glue;
                        };

                        return ret.empty() ? ret : ret.substr(0, ret.size() - glue.size());
                }

                ///
                /// \brief decode parameter by name: [name1=value1[,name2=value2[...]]
                /// the default value is returned if the parameter cannot be found or is invalid.
                ///
                template
                <
                        class tvalue
                >
                tvalue from_params(const std::string& params, const std::string& param_name, tvalue default_value)
                {
                        const auto tokens = text::split(params, ",");
                        for (std::size_t i = 0; i < tokens.size(); i ++)
                        {
                                const auto dual = text::split(tokens[i], "=");
                                if (dual.size() == 2 && dual[0] == param_name)
                                {
                                        std::string value = dual[1];
                                        for (   std::size_t j = i + 1;
                                                j < tokens.size() && tokens[j].find("=") == std::string::npos;
                                                j ++)
                                        {
                                                value += "," + tokens[j];
                                        }

                                        try
                                        {
                                                return from_string<tvalue>(value);
                                        }
                                        catch (std::exception&)
                                        {
                                                return default_value;
                                        }
                                }
                        }

                        return default_value;
                }
        }
}

