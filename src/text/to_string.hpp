#pragma once

#include <typeinfo>
#include "enum_string.hpp"

namespace text
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
                        static std::string dispatch(int value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<long>
                {
                        static std::string dispatch(long value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<long long>
                {
                        static std::string dispatch(long long value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<unsigned int>
                {
                        static std::string dispatch(unsigned int value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<unsigned long>
                {
                        static std::string dispatch(unsigned long value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<unsigned long long>
                {
                        static std::string dispatch(unsigned long long value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<float>
                {
                        static std::string dispatch(float value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<double>
                {
                        static std::string dispatch(double value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<long double>
                {
                        static std::string dispatch(long double value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<std::string>
                {
                        static std::string dispatch(std::string value)
                        {
                                return value;
                        }
                };

                template <>
                struct to_string_t<const char*>
                {
                        static std::string dispatch(const char* value)
                        {
                                return value;
                        }
                };

                ///
                /// \brief cast enums to strings
                ///
                template
                <
                        typename tvalue
                >
                struct to_string_t<tvalue, typename std::enable_if<std::is_enum<tvalue>::value>::type>
                {
                        static std::string dispatch(tvalue value)
                        {
                                const auto vm = enum_string<tvalue>();
                                const auto it = vm.find(value);

                                return (it == vm.end()) ? "???" : it->second;
                        }
                };
        }

        template
        <
                typename tvalue
        >
        std::string to_string(tvalue value)
        {
                return detail::to_string_t<tvalue>::dispatch(value);
        }
}

