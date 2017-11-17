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
        /// \brief comparison operators.
        ///
        class json_reader_t;
        bool operator==(const json_reader_t& reader1, const json_reader_t& reader2);
        bool operator!=(const json_reader_t& reader1, const json_reader_t& reader2);

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
                json_reader_t(const string_t& str, const size_t pos = 0) :
                        m_str(str), m_pos(pos)
                {
                        next();
                }

                ///
                /// \brief returns the [begin, end) range of json tokens
                ///
                auto begin() const { return json_reader_t(m_str, 0); }
                auto end() const { return json_reader_t(m_str, m_str.size()); }

                ///
                /// \brief retrieve the json token at the current position
                ///
                token_t operator*() const
                {
                        return {substr(), strlen(), m_tag};
                }

                ///
                /// \brief move to the next token
                ///
                json_reader_t& operator++()
                {
                        return next();
                }

                json_reader_t operator++(int)
                {
                        const auto copy = *this;
                        next();
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

                        for (auto itend = end(); *this != itend; ++ *this)
                        {
                                switch (m_tag)
                                {
                                case json_tag::end_object:
                                        // done!
                                        return next();

                                case json_tag::name:
                                        last_name = substr();
                                        last_size = strlen();
                                        break;

                                case json_tag::value:
                                        set_pair(last_name, last_size, substr(), strlen(), nvs...);
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
                auto tag() const { return m_tag; }
                const auto& str() const { return m_str; }
                const auto& range() const { return m_range; }

        private:

                static const char* null() { return "null"; }
                static const char* tokens() { return "{}[],:"; }
                static const char* spaces() { return " \t\n\r"; }

                static bool is_space(const char c)
                {
                        assert(::strlen(spaces()) == 4);
                        return spaces()[0] == c || spaces()[1] == c || spaces()[2] == c || spaces()[3] == c;
                }

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
                        while (begin < end && m_str.size() && is_space(m_str[begin])) ++ begin;
                        while (end > begin && is_space(m_str[end - 1])) -- end;

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

                json_reader_t& next()
                {
                        skip(spaces());
                        auto prev_pos = m_pos;
                        m_tag = json_tag::none;
                        while (find(tokens()))
                        {
                                m_range = trim(prev_pos, m_pos);
                                const auto some = strlen() > 0;

                                switch (m_str[m_pos])
                                {
                                case '{': m_tag = json_tag::new_object; break;
                                case '}': m_tag = some ? json_tag::value : json_tag::end_object; if (some) -- m_pos; break;
                                case '[': m_tag = json_tag::new_array; break;
                                case ']': m_tag = some ? json_tag::value : json_tag::end_array; if (some) -- m_pos; break;
                                case ',': m_tag = some ? json_tag::value : json_tag::none; break;
                                case ':': m_tag = json_tag::name; break;
                                default:  assert(false);
                                }

                                prev_pos = ++ m_pos;

                                if (m_tag == json_tag::value && !m_str.compare(m_range.first, strlen(), null()))
                                {
                                        m_tag = json_tag::null;
                                }
                                if (m_tag != json_tag::none)
                                {
                                        break;
                                }
                        }
                        skip(spaces());

                        if (m_pos == string_t::npos)
                        {
                                m_pos = m_str.size();
                        }

                        return *this;
                }

                static void set_pair(const char*, const size_t, const char*, const size_t)
                {
                }

                template <typename tvalue, typename... tnames_and_values>
                static void set_pair(
                        const char* name, const size_t name_size,
                        const char* value, const size_t value_size,
                        const char* object_name, tvalue& object_value,
                        tnames_and_values&... nvs)
                {
                        if (strncmp(object_name, name, std::min(::strlen(object_name), name_size)) == 0)
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
                json_tag        m_tag{json_tag::none};
        };

        inline bool operator==(const json_reader_t& reader1, const json_reader_t& reader2)
        {
                return  (&reader1.str() == &reader2.str() || reader1.str() == reader2.str()) &&
                        reader1.pos() == reader2.pos() &&
                        reader1.tag() == reader2.tag();
        }
        inline bool operator!=(const json_reader_t& reader1, const json_reader_t& reader2)
        {
                return  (&reader1.str() != &reader2.str() && reader1.str() != reader2.str())  ||
                        reader1.pos() != reader2.pos() ||
                        reader1.tag() != reader2.tag();
        }
}
