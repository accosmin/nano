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
        tvalue from_params(const string_t& params, const string_t& param_name, tvalue default_value)
        {
                auto begin = params.find(param_name + "=");
                if (begin == string_t::npos)
                {
                        return default_value;
                }
                else
                {
                        begin += param_name.size() + 1;
                        const auto end = params.find(",", begin);
                        const auto size = (end == string_t::npos ? params.size() : end) - begin;
                        const auto value = params.substr(begin, size);
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
}

