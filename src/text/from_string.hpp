#pragma once

#include <typeinfo>
#include <stdexcept>
#include "algorithm.h"
#include "enum_string.hpp"

namespace text
{
        namespace detail
        {
                template <typename, typename = void>
                struct from_string_t;

                ///
                /// \brief cast strings to built-int types
                ///
                template <>
                struct from_string_t<short>
                {
                        static short dispatch(const std::string& str)
                        {
                                return static_cast<short>(std::stoi(str));
                        }
                };

                template <>
                struct from_string_t<int>
                {
                        static int dispatch(const std::string& str)
                        {
                                return std::stoi(str);
                        }
                };

                template <>
                struct from_string_t<long>
                {
                        static long dispatch(const std::string& str)
                        {
                                return std::stol(str);
                        }
                };

                template <>
                struct from_string_t<long long>
                {
                        static long long dispatch(const std::string& str)
                        {
                                return std::stoll(str);
                        }
                };

                template <>
                struct from_string_t<unsigned long>
                {
                        static unsigned long dispatch(const std::string& str)
                        {
                                return std::stoul(str);
                        }
                };

                template <>
                struct from_string_t<unsigned long long>
                {
                        static unsigned long long dispatch(const std::string& str)
                        {
                                return std::stoull(str);
                        }
                };

                template <>
                struct from_string_t<float>
                {
                        static float dispatch(const std::string& str)
                        {
                                return std::stof(str);
                        }
                };

                template <>
                struct from_string_t<double>
                {
                        static double dispatch(const std::string& str)
                        {
                                return std::stod(str);
                        }
                };

                template <>
                struct from_string_t<long double>
                {
                        static long double dispatch(const std::string& str)
                        {
                                return std::stold(str);
                        }
                };

                template <>
                struct from_string_t<std::string>
                {
                        static std::string dispatch(const std::string& str)
                        {
                                return str;
                        }
                };

                ///
                /// \brief cast strings to enums
                ///
                template
                <
                        typename tvalue
                >
                struct from_string_t<tvalue, typename std::enable_if<std::is_enum<tvalue>::value>::type>
                {
                        static tvalue dispatch(const std::string& str)
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
                };
        }

        template
        <
                typename tvalue
        >
        tvalue from_string(const std::string& str)
        {
                return detail::from_string_t<tvalue>::dispatch(str);
        }

        ///
        /// \brief construct an operator to compare two strings numerically
        ///
        template
        <
                typename tscalar
        >
        auto make_less_from_string()
        {
                return [] (const std::string& v1, const std::string& v2)
                {
                        return from_string<tscalar>(v1) < from_string<tscalar>(v2);
                };
        }

        ///
        /// \brief construct an operator to compare two strings numerically
        ///
        template
        <
                typename tscalar
        >
        auto make_greater_from_string()
        {
                return [] (const std::string& v1, const std::string& v2)
                {
                        return from_string<tscalar>(v1) > from_string<tscalar>(v2);
                };
        }
}

