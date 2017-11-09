#pragma once

#include "cast.h"

namespace nano
{
        ///
        /// \brief limited ascii-based JSON reader.
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
                                const auto curr_pos = m_pos;
                                const auto curr_range = trim(prev_pos, curr_pos);

                                if (curr_range.first < curr_range.second)
                                {
                                        const auto begin = curr_range.first;
                                        const auto end = curr_range.second;
                                        std::cout << "token = [" << m_text.substr(begin, end - begin) << "]" << std::endl;
                                        ranges.push_back(curr_range);
                                }

                                /*
                                switch (m_text[m_pos])
                                {
                                case '{':       callback(m_text, curr, ++ m_pos, tag::begin_object); break;
                                case '}':       callback(m_text, curr, ++ m_pos, tag::end_object); break;
                                case '[':       callback(m_text, curr, ++ m_pos, tag::begin_array); break;
                                case ']':       callback(m_text, curr, ++ m_pos, tag::end_array); break;
                                case ',':       callback(m_text, curr, ++ m_pos, tag::next); break;
                                case ':':       callback(m_text, curr, ++ m_pos, tag::pair); break;
                                default:        assert(false);
                                }*/

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

                // attributes
                const string_t& m_text;
                size_t          m_pos{0};
        };
}
