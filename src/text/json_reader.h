#pragma once

#include "cast.h"

namespace nano
{
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
                        typename tcallback_begin_object,        ///< (const char* name, size)
                        typename tcallback_end_object,          ///< ()
                        typename tcallback_begin_array,         ///< (const char* name, size)
                        typename tcallback_end_array,           ///< ()
                        typename tcallback_null,                ///< (const char* name, size)
                        typename tcallback_value                ///< (const char* name, size, const char* value, size)
                >
                void parse(
                        const tcallback_begin_object& callback_begin_object,
                        const tcallback_end_object& callback_end_object,
                        const tcallback_begin_array& callback_begin_array,
                        const tcallback_end_array& callback_end_array,
                        const tcallback_null& callback_null,
                        const tcallback_value& callback_value)
                {
                        skip(spaces());

                        auto prev_pos = m_pos;

                        std::vector<range_t> ranges;
                        ranges.emplace_back(0, 0);

                        while (find(tokens()))
                        {
                                const auto range = trim(prev_pos, m_pos);
                                if (range.first < range.second)
                                {
                                        ranges.push_back(range);
                                }

                                switch (m_text[m_pos])
                                {
                                case '{':
                                        handle1(ranges, callback_begin_object);
                                        ranges.clear();
                                        break;

                                case '}':
                                        handle2(ranges, callback_null, callback_value);
                                        callback_end_object();
                                        ranges.clear();
                                        break;

                                case '[':
                                        handle1(ranges, callback_begin_array);
                                        ranges.clear();
                                        break;

                                case ']':
                                        handle2(ranges, callback_null, callback_value);
                                        callback_end_array();
                                        ranges.clear();
                                        break;

                                case ',':
                                        handle2(ranges, callback_null, callback_value);
                                        break;

                                case ':':
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
                void handle1(const std::vector<range_t>& ranges, const tcallback& callback) const
                {
                        if (!ranges.empty())
                        {
                                const auto& r = *ranges.rbegin();
                                callback(substr(r), strlen(r));
                        }
                }

                template <typename tcallback_null, typename tcallback_value>
                void handle2(const std::vector<range_t>& ranges,
                        const tcallback_null& callback_null,
                        const tcallback_value& callback_value) const
                {
                        if (ranges.size() >= 2)
                        {
                                const auto& r1 = ranges[ranges.size() - 2];
                                const auto& r2 = ranges[ranges.size() - 1];

                                if (m_text.compare(r2.first, r2.second - r2.first, null()) == 0)
                                {
                                        callback_null(substr(r1), strlen(r1));
                                }
                                else
                                {
                                        callback_value(substr(r1), strlen(r1), substr(r2), strlen(r2));
                                }
                        }
                }

                // attributes
                const string_t& m_text;
                size_t          m_pos{0};
        };
}
