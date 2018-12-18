#pragma once

#include <limits>
#include <utility>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include "stringi.h"

namespace nano
{
        template <typename tenum>
        using enum_map_t = std::vector<std::pair<tenum, string_t>>;

        ///
        /// \brief maps all possible values of an enum to string.
        /// NB: to specialize it, such that nano::to_string & nano::from_string work on a particular enum
        ///
        template <typename tenum>
        enum_map_t<tenum> enum_string();

        ///
        /// \brief collect all the values for an enum type.
        ///
        template <typename tenum>
        std::vector<tenum> enum_values()
        {
                static const auto mapping = enum_string<tenum>();
                std::vector<tenum> enums(mapping.size());
                for (const auto& elem : mapping)
                {
                        enums.push_back(elem.first);
                }
                return enums;
        }

        ///
        /// \brief cast built-in types to strings
        ///
        template <typename tvalue, typename = void>
        struct to_string_t
        {
                static string_t cast(const tvalue value)
                {
                        std::ostringstream ss;
                        ss.precision(16);
                        ss << value;
                        return ss.str();
                }
        };

        template <>
        struct to_string_t<string_t, void>
        {
                static string_t cast(const string_t& value)
                {
                        return value;
                }
        };

        template <>
        struct to_string_t<const char*, void>
        {
                static string_t cast(const char* value)
                {
                        return value;
                }
        };

        ///
        /// \brief cast enums to strings
        ///
        template <typename tenum>
        struct to_string_t<tenum, typename std::enable_if<std::is_enum<tenum>::value>::type>
        {
                static string_t cast(const tenum value)
                {
                        static const auto mapping = enum_string<tenum>();
                        for (const auto& elem : mapping)
                        {
                                if (elem.first == value)
                                {
                                        return elem.second;
                                }
                        }
                        const auto str = std::to_string(static_cast<int>(value));
                        const auto msg = string_t("missing mapping for enumeration ") + typeid(tenum).name() + " <" + str + ">!";
                        throw std::invalid_argument(msg);
                }
        };

        ///
        /// \brief cast values to string.
        ///
        template <typename tvalue>
        string_t to_string(const tvalue value)
        {
                /// todo: replace this with "if constepr" in c++17
                return to_string_t<tvalue>::cast(value);
        }

        template <typename, typename = void>
        struct from_string_t;

        ///
        /// \brief cast strings to builtin types
        ///
        template <>
        struct from_string_t<short>
        {
                static short cast(const string_t& str)
                {
                        return static_cast<short>(std::stoi(str));
                }
        };

        template <>
        struct from_string_t<int>
        {
                static int cast(const string_t& str)
                {
                        return std::stoi(str);
                }
        };

        template <>
        struct from_string_t<long>
        {
                static long cast(const string_t& str)
                {
                        return std::stol(str);
                }
        };

        template <>
        struct from_string_t<long long>
        {
                static long long cast(const string_t& str)
                {
                        return std::stoll(str);
                }
        };

        template <>
        struct from_string_t<unsigned long>
        {
                static unsigned long cast(const string_t& str)
                {
                        return std::stoul(str);
                }
        };

        template <>
        struct from_string_t<unsigned long long>
        {
                static unsigned long long cast(const string_t& str)
                {
                        return std::stoull(str);
                }
        };

        template <>
        struct from_string_t<float>
        {
                static float cast(const string_t& str)
                {
                        return std::stof(str);
                }
        };

        template <>
        struct from_string_t<double>
        {
                static double cast(const string_t& str)
                {
                        return std::stod(str);
                }
        };

        template <>
        struct from_string_t<long double>
        {
                static long double cast(const string_t& str)
                {
                        return std::stold(str);
                }
        };

        template <>
        struct from_string_t<string_t>
        {
                static string_t cast(const string_t& str)
                {
                        return str;
                }
        };

        ///
        /// \brief cast strings to enums
        ///
        template <typename tenum>
        struct from_string_t<tenum, typename std::enable_if<std::is_enum<tenum>::value>::type>
        {
                static tenum cast(const string_t& str)
                {
                        for (const auto& elem : enum_string<tenum>())
                        {
                                if (elem.second == str)
                                {
                                        return elem.first;
                                }
                        }
                        const auto msg = string_t("invalid ") + typeid(tenum).name() + " <" + str + ">!";
                        throw std::invalid_argument(msg);
                }
        };

        ///
        /// \brief cast string to values.
        ///
        template <typename tvalue>
        tvalue from_string(const string_t& str)
        {
                /// todo: replace this with "if constexpr" in c++17
                return from_string_t<tvalue>::cast(str);
        }

        ///
        /// \brief cast string to values and use the given default value if casting fails.
        ///
        template <typename tvalue>
        tvalue from_string(const string_t& str, const tvalue& default_value)
        {
                try
                {
                        return from_string<tvalue>(str);
                }
                catch (std::exception&)
                {
                        return default_value;
                }
        }

        ///
        /// \brief construct an operator to compare two strings numerically
        ///
        template <typename tscalar>
        auto make_less_from_string()
        {
                return [] (const string_t& v1, const string_t& v2)
                {
                        return  from_string<tscalar>(v1, std::numeric_limits<tscalar>::lowest()) <
                                from_string<tscalar>(v2, std::numeric_limits<tscalar>::max());
                };
        }

        ///
        /// \brief construct an operator to compare two strings numerically
        ///
        template <typename tscalar>
        auto make_greater_from_string()
        {
                return [] (const string_t& v1, const string_t& v2)
                {
                        return  from_string<tscalar>(v1, std::numeric_limits<tscalar>::max()) >
                                from_string<tscalar>(v2, std::numeric_limits<tscalar>::lowest());
                };
        }

        namespace detail
        {
                template <typename tvalue>
                void strcat(string_t& str, const tvalue& value)
                {
                        str += to_string(value);
                }

                template <>
                inline void strcat<string_t>(string_t& str, const string_t& value)
                {
                        str += value;
                }

                template <>
                inline void strcat<char>(string_t& str, const char& value)
                {
                        str += value;
                }

                inline void strcat(string_t& str, const char* value)
                {
                        str += value;
                }

                template <typename tvalue, typename... tvalues>
                void strcat(string_t& str, const tvalue& value, const tvalues&... values)
                {
                        strcat(str, value);
                        strcat(str, values...);
                }
        }

        ///
        /// \brief concatenate a list of potentially heterogeneous values into a string
        ///
        template <typename... tvalues>
        string_t strcat(const tvalues&... values)
        {
                string_t str;
                detail::strcat(str, values...);
                return str;
        }

        ///
        /// \brief compact a list of values into a string using the given "glue" string.
        ///
        template <typename titerator>
        string_t join(titerator begin, const titerator end, const char* glue = ",",
                const char* prefix = "[", const char* suffix = "]")
        {
                string_t ret;
                if (prefix)
                {
                        ret += prefix;
                }
                for (; begin != end; )
                {
                        detail::strcat(ret, *begin);
                        if (++ begin != end)
                        {
                                ret += glue;
                        }
                }
                if (suffix)
                {
                        ret += suffix;
                }

                return ret;
        }

        template <typename tcontainer>
        string_t join(const tcontainer& values, const char* glue = ",",
                const char* prefix = "[", const char* suffix = "]")
        {
                return join(values.begin(), values.end(), glue, prefix, suffix);
        }
}
