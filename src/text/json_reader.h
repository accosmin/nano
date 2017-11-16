#pragma once

#include "cast.h"
#include <cassert>

#include <iostream>

namespace nano
{
        ///
        /// \brief JSON tags.
        ///
        enum class json_tag
        {
                new_object,
                end_object,
                new_array,
                end_array,
                name,
                null,
                value,
                none
        };

        template <>
        inline enum_map_t<json_tag> enum_string<json_tag>()
        {
                return
                {
                        { json_tag::new_object, "new_object" },
                        { json_tag::end_object, "end_object" },
                        { json_tag::new_array, "new_array" },
                        { json_tag::end_array, "end_array" },
                        { json_tag::name, "name" },
                        { json_tag::null, "null" },
                        { json_tag::value, "value" },
                        { json_tag::none, "none" }
                };
        }

        ///
        /// \brief limited ascii-based JSON reader (e.g. no support for escaping special characters).
        ///     the decoding is implemented ala STL iterators:
        ///             for (const auto& token : json_reader_t(text))
        ///             {
        ///                     print(token); /// tuple of <text*, size, json_tag>
        ///             }
        ///
        class json_reader_t
        {
        public:

                using range_t = std::pair<size_t, size_t>;
                using token_t = std::tuple<const char*, size_t, json_tag>;

                ///
                /// \brief constructor
                ///
                json_reader_t(const string_t& str, const size_t pos = 0, const range_t& range = {0, 0}) :
                        m_str(str), m_pos(pos), m_range(range)
                {
                }

                ///
                /// \brief returns the [begin, end) range of json tokens
                ///
                auto begin() const { return json_reader_t(m_str, m_pos, m_range); }
                auto end() const { return json_reader_t(m_str, m_str.size(), m_range); }

                ///
                /// \brief retrieve the json token at the current position
                ///
                token_t operator*()
                {
                        while (m_pos < m_str.size())
                        {
                                std::cout << "::operator*: token = " << string_t(substr(), strlen())
                                          << ", pos = " << m_pos << "/" << m_str.size() << std::endl;
                                switch (m_str[m_pos])
                                {
                                case '{': return handle0(json_tag::new_object);
                                case '}': return strlen() ? handle1(json_tag::value) : handle0(json_tag::end_object);
                                case '[': return handle0(json_tag::new_array);
                                case ']': return strlen() ? handle1(json_tag::value) : handle0(json_tag::end_array);
                                case ',': return handle1(json_tag::value);
                                case ':': return handle1(json_tag::name);
                                default:  ++*this; break;
                                }
                        }
                        return {nullptr, 0, json_tag::none};
                }

                ///
                /// \brief move to the next token
                ///
                json_reader_t& operator++()
                {
                        skip(spaces());
                        auto prev_pos = m_pos;
                        while (find(tokens()))
                        {
                                std::cout << "::operator++: token = " << string_t(substr(), strlen())
                                        << ", pos = " << m_pos << "/" << m_str.size() << std::endl;
                                m_range = trim(prev_pos, m_pos);
                                /*switch (m_str[m_pos])
                                {
                                case '{': ++ m_pos; return *this;
                                case '}': return strlen() ? handle1(json_tag::value) : handle0(json_tag::end_object);
                                case '[': return handle0(json_tag::new_array);
                                case ']': return strlen() ? handle1(json_tag::value) : handle0(json_tag::end_array);
                                case ',': return handle1(json_tag::value);
                                case ':': return handle1(json_tag::name);
                                default:  assert(false);
                                }*/
                                prev_pos = ++ m_pos;
                                if (strlen() > 0)
                                {
                                        return *this;
                                }
                        }

                        m_pos = m_str.size();
                        return *this;
                }

                json_reader_t operator++(int)
                {
                        const auto copy = *this;
                        ++ *this;
                        return copy;
                }

                ///
                /// \brief read a set of name:value pairs that compose a JSON object.
                ///
                template <typename... tnames_and_values>
                json_reader_t& object(tnames_and_values&... nvs)
                {
                        const char* last_name = nullptr;
                        size_t last_size = 0;

                        for (const auto& token : *this)
                        {
                                const auto name = std::get<0>(token);
                                const auto size = std::get<1>(token);
                                const auto jtag = std::get<2>(token);

                                std::cout << "token: " << string_t(name, size) << ", tag = " << to_string(jtag) << std::endl;

                                switch (jtag)
                                {
                                case json_tag::end_object:
                                        // done!
                                        return *this;

                                case json_tag::name:
                                        last_name = name;
                                        last_size = size;
                                        break;

                                case json_tag::value:
                                        set_pair(last_name, last_size, name, size, nvs...);
                                        break;

                                default:
                                        // continue
                                        break;
                                }
                        }

                        return *this;
                }

                ///
                /// \brief access functions
                ///
                auto pos() const { return m_pos; }
                const auto& str() const { return m_str; }
                const auto& range() const { return m_range; }

        private:

                static const char* null() { return "null"; }
                static const char* tokens() { return "{}[],:"; }
                static const char* spaces() { return " \t\n\r"; }

                bool find(const char* str)
                {
                        return (m_pos = m_str.find_first_of(str, m_pos)) != string_t::npos;
                }

                bool skip(const char* str)
                {
                        return (m_pos = m_str.find_first_not_of(str, m_pos)) != string_t::npos;
                }

                range_t trim(size_t begin, size_t end) const
                {
                        begin = m_str.find_first_not_of(spaces(), begin);
                        end = std::min(end, m_str.find_first_of(spaces(), begin));

                        if (begin < end)
                        {
                                if (m_str[begin] == '\"') ++ begin;
                                if (m_str[end - 1] == '\"') --end;
                        }

                        return {begin, end};
                }

                const char* substr() const
                {
                        assert(m_range.first < m_str.size());
                        return &m_str[m_range.first];
                }

                size_t strlen() const
                {
                        assert(m_range.first <= m_range.second);
                        assert(m_range.second <= m_str.size());
                        return m_range.second - m_range.first;
                }

                token_t handle0(const json_tag tag) const
                {
                        return {substr(), strlen(), tag};
                }

                token_t handle1(const json_tag tag) const
                {
                        assert(m_range.first < m_range.second);
                        const auto none = (tag == json_tag::value) && !m_str.compare(m_range.first, strlen(), null());
                        return {substr(), strlen(), none ? json_tag::null : tag};
                }

                static void set_pair(const char*, const size_t, const char*, const size_t)
                {
                }

                template <typename tname, typename tvalue, typename... tnames_and_values>
                static void set_pair(
                        const char* name, const size_t name_size,
                        const char* value, const size_t value_size,
                        const tname& object_name, tvalue& object_value,
                        tnames_and_values&... nvs)
                {
                        // todo: check if possible to do the comparison without creating a string copy
                        if (string_t(object_name) == string_t(name, name_size))
                        {
                                object_value = from_string<tvalue>(string_t(value, value_size));
                        }
                        else
                        {
                                set_pair(name, name_size, value, value_size, nvs...);
                        }
                }

                // attributes
                const string_t& m_str;
                size_t          m_pos{0};
                range_t         m_range{0, 0};
        };

        ///
        /// \brief comparison operators.
        ///
        inline bool operator==(const json_reader_t& reader1, const json_reader_t& reader2)
        {
                return  reader1.str() == reader2.str() &&
                        reader1.pos() == reader2.pos();
        }
        inline bool operator!=(const json_reader_t& reader1, const json_reader_t& reader2)
        {
                return  reader1.str() != reader2.str() ||
                        reader1.pos() != reader2.pos();
        }
}
