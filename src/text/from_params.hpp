#pragma once

#include "from_string.hpp"

namespace nano
{
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
                const auto tokens = nano::split(params, ",");
                for (std::size_t i = 0; i < tokens.size(); i ++)
                {
                        const auto dual = nano::split(tokens[i], "=");
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

