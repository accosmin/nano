#pragma once

#include "enum_string.hpp"

namespace nano
{
        namespace detail
        {
                template <typename, typename = void>
                struct to_string_t;

                ///
                /// \brief cast built-in types to strings
                ///
                template <>
                struct to_string_t<int>
                {
                        static string_t dispatch(int value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<long>
                {
                        static string_t dispatch(long value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<long long>
                {
                        static string_t dispatch(long long value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<unsigned int>
                {
                        static string_t dispatch(unsigned int value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<unsigned long>
                {
                        static string_t dispatch(unsigned long value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<unsigned long long>
                {
                        static string_t dispatch(unsigned long long value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<float>
                {
                        static string_t dispatch(float value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<double>
                {
                        static string_t dispatch(double value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<long double>
                {
                        static string_t dispatch(long double value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<string_t>
                {
                        static string_t dispatch(string_t value)
                        {
                                return value;
                        }
                };

                template <>
                struct to_string_t<const char*>
                {
                        static string_t dispatch(const char* value)
                        {
                                return value;
                        }
                };

                ///
                /// \brief cast enums to strings
                ///
                template <typename tvalue>
                struct to_string_t<tvalue, typename std::enable_if<std::is_enum<tvalue>::value>::type>
                {
                        static string_t dispatch(tvalue value)
                        {
                                const auto vm = enum_string<tvalue>();
                                const auto it = vm.find(value);

                                return (it == vm.end()) ? "???" : it->second;
                        }
                };
        }

        template <typename tvalue>
        string_t to_string(tvalue value)
        {
                return detail::to_string_t<tvalue>::dispatch(value);
        }
}

