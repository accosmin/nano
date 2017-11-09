#pragma once

#include "cast.h"

namespace nano
{
        ///
        /// \brief JSON tags.
        ///
        enum class json_tag
        {
                begin_object,
                end_object,
                begin_array,
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
                        { json_tag::begin_object, "begin_object" },
                        { json_tag::end_object, "end_object" },
                        { json_tag::begin_array, "begin_array" },
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

                json_reader_t(const string_t& text) :
                        m_text(text)
                {
                }

                template
                <
                        typename tcallback      ///< (const char* name, size, tag type)
                >
                void parse(const tcallback& callback)
                {
                        skip(spaces());

                        auto prev_pos = m_pos;
                        while (find(tokens()))
                        {
                                const auto range = trim(prev_pos, m_pos);

                                switch (m_text[m_pos])
                                {
                                case '{':
                                        handle0(callback, json_tag::begin_object);
                                        break;

                                case '}':
                                        handle1(range, callback, json_tag::value);
                                        handle0(callback, json_tag::end_object);
                                        break;

                                case '[':
                                        handle0(callback, json_tag::begin_array);
                                        break;

                                case ']':
                                        handle1(range, callback, json_tag::value);
                                        handle0(callback, json_tag::end_array);
                                        break;

                                case ',':
                                        handle1(range, callback, json_tag::value);
                                        break;

                                case ':':
                                        handle1(range, callback, json_tag::name);
                                        break;

                                default:
                                        assert(false);
                                }

                                prev_pos = ++ m_pos;
                                skip(spaces());
                        }
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
                void handle0(const tcallback& callback, const json_tag tag) const
                {
                        callback(&m_text[m_pos], 0, tag);
                }

                template <typename tcallback>
                void handle1(const range_t& range, const tcallback& callback, const json_tag tag) const
                {
                        if (range.first < range.second)
                        {
                                const auto is_null =
                                        tag == json_tag::value &&
                                        m_text.compare(range.first, range.second - range.first, null()) == 0;
                                callback(substr(range), strlen(range), is_null ? json_tag::null : tag);
                        }
                }

                // attributes
                const string_t& m_text;
                size_t          m_pos{0};
        };
}
