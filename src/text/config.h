#pragma once

#include "to_string.h"
#include "from_string.h"

namespace nano
{
        ///
        /// \brief encode parameter by name: [name1=value1[,name2=value2[...]]
        ///
        template <typename tvalue>
        string_t to_params(const char* name, const tvalue& value)
        {
                return string_t(name) + "=" + to_string(value);
        }

        template <typename tvalue, typename... tvalues>
        string_t to_params(const char* name, const tvalue& value, const tvalues&... values)
        {
                return to_params(name, value) + "," + to_params(values...);
        }

        template <typename tvalue, typename... tvalues>
        string_t to_params(const string_t& params, const char* name, const tvalue& value, const tvalues&... values)
        {
                return params + (params.empty() ? "" : ",") + to_params(name, value, values...);
        }

        ///
        /// \brief computes the [begin, begin + size) range of the value associated for the given parameter name.
        /// \return true if there is any value associated.
        ///
        inline bool value_range(const string_t& params, const string_t& param_name, size_t& begin, size_t& size)
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
                        return (size = (end == string_t::npos ? params.size() : end) - begin) > 0;
                }
        }

        ///
        /// \brief decode parameter by name: [name1=value1[\[description\]][,name2=value2[...]]
        /// the default value is returned if the parameter cannot be found or is invalid.
        ///
        template <typename tvalue>
        tvalue from_params(const string_t& params, const string_t& param_name, tvalue default_value)
        {
                try
                {
                        size_t begin, size;
                        return  value_range(params, param_name, begin, size) ?
                                from_string<tvalue>(params.substr(begin, size)) :
                                default_value;
                }
                catch (std::exception&)
                {
                        return default_value;
                }
        }

        ///
        /// \brief decode parameter by name: [name1=value1[\[description\]][,name2=value2[...]]
        /// an exception is thrown if the parameter cannot be found or is invalid.
        ///
        template <typename tvalue>
        tvalue from_params(const string_t& params, const string_t& param_name)
        {
                size_t begin, size;
                if (value_range(params, param_name, begin, size))
                {
                        return from_string<tvalue>(params.substr(begin, size));
                }
                throw std::runtime_error("invalid value for parameter <" + param_name + ">");
        }
}
