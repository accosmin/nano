#pragma once

#include <typeinfo>
#include <stdexcept>
#include "equals.hpp"
#include "enum_string.hpp"

namespace ncv
{
        namespace text
        {
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
                /// \brief from_string specialization for enums
                ///
                template
                <
                        typename tvalue
                >
                tvalue from_string(const std::string& str)
                {
                        const auto vm = enum_string<tvalue>();
                        const auto it = std::find_if(vm.begin(), vm.end(), [&str] (const auto& v)
                        {
                                return text::iequals(str, v.second);
                        });

                        if (it == vm.end())
                        {
                                const auto msg = std::string("invalid ") + typeid(tvalue).name() + " <" + str + ">!";
                                throw std::invalid_argument(msg);
                        }
                        return it->first;
                }
        }
}

