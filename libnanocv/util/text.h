#pragma once

#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

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
                using namespace boost::algorithm;

                ///
                /// \brief align a string to fill the given size
                ///
                std::string resize(const std::string& str, std::size_t size,
                                   align alignment = align::left, char fill_char = ' ');

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
                tvalue from_string(const std::string& str)
                {
                        return boost::lexical_cast<tvalue>(str);
                }

                ///
                /// \brief compact a list of values into a string using the given "glue" string
                ///
                template
                <
                        typename tvalue
                >
                std::string concatenate(const std::vector<tvalue>& values, const std::string& glue = ",")
                {
                        std::string ret;
                        std::for_each(std::begin(values), std::end(values), [&] (const tvalue& val)
                        {
                                ret += to_string(val) + glue;
                        });

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
                        std::vector<std::string> tokens, dual;

                        text::split(tokens, params, text::is_any_of(","));
                        for (std::size_t i = 0; i < tokens.size(); i ++)
                        {
                                text::split(dual, tokens[i], text::is_any_of("="));
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

