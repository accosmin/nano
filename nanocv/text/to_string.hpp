#pragma once

#include <typeinfo>
#include "enum_string.hpp"

namespace ncv
{
        namespace text
        {
                ///
                /// \brief cast to string for built-in types
                ///
                template
                <
                        typename tvalue
                >
                std::string to_string(tvalue value);

                template <>
                inline std::string to_string<int>(int value)
                {
                        return std::to_string(value);
                }

                template <>
                inline std::string to_string<long>(long value)
                {
                        return std::to_string(value);
                }

                template <>
                inline std::string to_string<long long>(long long value)
                {
                        return std::to_string(value);
                }

                template <>
                inline std::string to_string<unsigned int>(unsigned int value)
                {
                        return std::to_string(value);
                }

                template <>
                inline std::string to_string<unsigned long>(unsigned long value)
                {
                        return std::to_string(value);
                }

                template <>
                inline std::string to_string<unsigned long long>(unsigned long long value)
                {
                        return std::to_string(value);
                }

                template <>
                inline std::string to_string<float>(float value)
                {
                        return std::to_string(value);
                }

                template <>
                inline std::string to_string<double>(double value)
                {
                        return std::to_string(value);
                }

                template <>
                inline std::string to_string<long double>(long double value)
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
                /// \brief to_string specialization for enums
                ///
                template
                <
                        typename tvalue
                >
                std::string to_string(tvalue value)
                {
                        typedef typename std::enable_if<std::is_enum<tvalue>::value>::type to_string_for_enums;

                        const auto vm = enum_string<tvalue>();
                        const auto it = vm.find(value);

                        return (it == vm.end()) ? "????" : it->second;
                }
        }
}

