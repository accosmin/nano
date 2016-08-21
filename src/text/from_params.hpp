#pragma once

#include "from_string.hpp"

namespace nano
{
        namespace detail
        {
                static bool value_range(const string_t& params, const string_t& param_name, size_t& begin, size_t& size)
                {
                        begin = params.find(param_name + "=");
                        if (begin == string_t::npos)
                        {
                                return false;
                        }
                        else
                        {
                                begin += param_name.size() + 1;
                                const auto end = std::min(params.find(",", begin), params.find("[", begin));
                                size = (end == string_t::npos ? params.size() : end) - begin;
                                return size > 0;
                        }
                }
        }

        ///
        /// \brief decode parameter by name: [name1=value1[\[description\]][,name2=value2[...]]
        /// the default value is returned if the parameter cannot be found or is invalid.
        ///
        template
        <
                class tvalue
        >
        tvalue from_params(const string_t& params, const string_t& param_name, tvalue default_value)
        {
                size_t begin, size;
                if (!detail::value_range(params, param_name, begin, size))
                {
                        return default_value;
                }
                else
                {
                        try
                        {
                                return from_string<tvalue>(params.substr(begin, size));
                        }
                        catch (std::exception&)
                        {
                                return default_value;
                        }
                }
        }

        ///
        /// \brief decode parameter by name: [name1=value1[\[description\]][,name2=value2[...]]
        /// an exception is thrown if the parameter cannot be found or is invalid.
        ///
        template
        <
                class tvalue
        >
        tvalue from_params(const string_t& params, const string_t& param_name)
        {
                size_t begin, size;
                if (detail::value_range(params, param_name, begin, size))
                {
                        return from_string<tvalue>(params.substr(begin, size));
                }
                throw std::runtime_error("invalid value for parameter <" + param_name + ">");
        }
}

