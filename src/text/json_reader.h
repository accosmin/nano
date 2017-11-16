#pragma once

#include "cast.h"
#include <cassert>

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
                value
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
                        { json_tag::value, "value" }
                };
        }

        ///
        /// \brief limited ascii-based JSON reader (e.g. no support for escaping special characters).
        ///
        class json_reader_t
        {
        public:

                using range_t = std::pair<size_t, size_t>;

                ///
                /// \brief constructor
                ///
                json_reader_t(const string_t& text) :
                        m_text(text)
                {
                }

                ///
                /// \brief parse the text and call with (const char* name, size, tag type) for each token
                /// NB: the callback returns true if the parsing should continue or false otherwise.
                ///
                template <typename tcallback>
                void parse(const tcallback& callback)
                {
                        skip(spaces());

                        auto prev_pos = m_pos;
                        auto done = false;
                        while (!done && find(tokens()))
                        {
                                const auto range = trim(prev_pos, m_pos);

                                done = false;
                                switch (m_text[m_pos])
                                {
                                case '{':
                                        done = !handle0(callback, json_tag::new_object);
                                        break;

                                case '}':
                                        done = !handle1(range, callback, json_tag::value) ||
                                               !handle0(callback, json_tag::end_object);
                                        break;

                                case '[':
                                        done = !handle0(callback, json_tag::new_array);
                                        break;

                                case ']':
                                        done = !handle1(range, callback, json_tag::value) ||
                                               !handle0(callback, json_tag::end_array);
                                        break;

                                case ',':
                                        done = !handle1(range, callback, json_tag::value);
                                        break;

                                case ':':
                                        done = !handle1(range, callback, json_tag::name);
                                        break;

                                default:
                                        assert(false);
                                }

                                prev_pos = ++ m_pos;
                                skip(spaces());
                        }
                }

                ///
                /// \brief read a set of name:value pairs that compose a JSON object.
                ///
                template <typename... tnames_and_values>
                json_reader_t& object(tnames_and_values&... nvs)
                {
                        const char* last_name = nullptr;
                        size_t last_size = 0;

                        const auto callback = [&] (const char* name, const size_t size, const json_tag tag)
                        {
                                switch (tag)
                                {
                                case json_tag::end_object:
                                        // done!
                                        return false;

                                case json_tag::name:
                                        last_name = name;
                                        last_size = size;
                                        return true;

                                case json_tag::value:
                                        set_pair(last_name, last_size, name, size, nvs...);
                                        return true;

                                default:
                                        // continue
                                        return true;
                                }
                        };

                        parse(callback);
                        return *this;
                }

        private:

                static const char* null() { return "null"; }
                static const char* tokens() { return "{}[],:"; }
                static const char* spaces() { return " \t\n\r"; }

                bool find(const char* str)
                {
                        return (m_pos = m_text.find_first_of(str, m_pos)) != string_t::npos;
                }

                bool skip(const char* str)
                {
                        return (m_pos = m_text.find_first_not_of(str, m_pos)) != string_t::npos;
                }

                range_t trim(size_t begin, size_t end) const
                {
                        begin = m_text.find_first_not_of(spaces(), begin);
                        end = std::min(end, m_text.find_first_of(spaces(), begin));

                        if (begin < end)
                        {
                                if (m_text[begin] == '\"') ++ begin;
                                if (m_text[end - 1] == '\"') --end;
                        }

                        return {begin, end};
                }

                const char* substr(const range_t& range) const
                {
                        assert(range.first < m_text.size());
                        return &m_text[range.first];
                }

                size_t strlen(const range_t& range) const
                {
                        assert(range.first <= range.second);
                        assert(range.second <= m_text.size());
                        return range.second - range.first;
                }

                template <typename tcallback>
                bool handle0(const tcallback& callback, const json_tag tag) const
                {
                        return callback(&m_text[m_pos], 0, tag);
                }

                template <typename tcallback>
                bool handle1(const range_t& range, const tcallback& callback, const json_tag tag) const
                {
                        if (range.first < range.second)
                        {
                                const auto is_null =
                                        tag == json_tag::value &&
                                        m_text.compare(range.first, range.second - range.first, null()) == 0;
                                return callback(substr(range), strlen(range), is_null ? json_tag::null : tag);
                        }
                        return true;
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
                const string_t& m_text;
                size_t          m_pos{0};
        };
}
