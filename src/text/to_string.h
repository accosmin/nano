#pragma once

#include <typeinfo>
#include <stdexcept>
#include "enum_string.h"

namespace nano
{
        namespace detail
        {
                ///
                /// \brief cast built-in types to strings
                ///
                template <typename tvalue, typename = void>
                struct to_string_t
                {
                        static string_t dispatch(const tvalue value)
                        {
                                return std::to_string(value);
                        }
                };

                template <>
                struct to_string_t<string_t>
                {
                        static string_t dispatch(const string_t value)
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
                        static string_t dispatch(const tvalue value)
                        {
                                static const auto vm = enum_string<tvalue>();
                                const auto it = vm.find(value);
                                if (it == vm.end())
                                {
                                        const auto str = std::to_string(static_cast<int>(value));
                                        const auto msg = string_t("missing mapping for enumeration ") + typeid(tvalue).name() + " <" + str + ">!";
                                        throw std::invalid_argument(msg);
                                }
                                return it->second;
                        }
                };
        }

        ///
        /// \brief cast values to string.
        ///
        template <typename tvalue>
        string_t to_string(const tvalue value)
        {
                /// todo: replace this with "if constepr" in c++17
                return detail::to_string_t<tvalue>::dispatch(value);
        }

        ///
        /// \brief compact a list of values into a string using the given "glue" string.
        ///
        template <typename titerator>
        string_t concatenate(titerator begin, const titerator end, const char* glue = ",")
        {
                string_t ret;
                for (; begin != end; )
                {
                        ret += to_string(*begin);
                        if (++ begin != end)
                        {
                                ret += glue;
                        }
                }

                return ret;
        }

        template <typename tcontainer>
        string_t concatenate(const tcontainer& values, const char* glue = ",")
        {
                return concatenate(values.begin(), values.end(), glue);
        }
}
