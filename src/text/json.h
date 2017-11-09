#pragma once

#include "cast.h"

namespace nano
{
        ///
        /// \brief limited ascii-based JSON writer.
        ///
        class json_writer_t
        {
        public:

                json_writer_t(const size_t tabsize = 4, const size_t spacing = 1, const bool newline = true) :
                        m_tabsize(tabsize), m_spacing(spacing), m_newline(newline)
                {
                }

                json_writer_t& name(const char* tag)
                {
                        quote(tag);
                        m_text += ':';
                        return *this;
                }

                template <typename tvalue>
                json_writer_t& value(const tvalue& val)
                {
                        m_text.append(to_string(val));
                        return *this;
                }

                json_writer_t& value(const char* str)
                {
                        return quote(str);
                }

                json_writer_t& value(const string_t& str)
                {
                        return quote(str);
                }

                template <typename tvalue>
                json_writer_t& pair(const char* tag, const tvalue& val)
                {
                        name(tag);
                        return value(val);
                }

                json_writer_t& next()
                {
                        m_text.append(1, ',');
                        return *this;
                }

                template <typename... tvalues>
                json_writer_t& array(const tvalues&... vals)
                {
                        begin_array();
                        values(vals...);
                        return end_array();
                }

                json_writer_t& begin_array() { return keyword('['); }
                json_writer_t& begin_object() { return keyword('{'); }

                json_writer_t& end_array() { return keyword(']'); }
                json_writer_t& end_object() { return keyword('}'); }

                const string_t& get() { return m_text; }

        private:

                json_writer_t& keyword(const char tag)
                {
                        m_text += tag;
                        return *this;
                }

                template <typename tstr>
                json_writer_t& quote(const tstr& str)
                {
                        m_text += '\"';
                        m_text += str;
                        m_text += '\"';
                        return *this;
                }

                template <typename tvalue>
                void values(const tvalue& val)
                {
                        value(val);
                }

                template <typename tvalue, typename... tvalues>
                void values(const tvalue& val, const tvalues&... vals)
                {
                        value(val);
                        if (sizeof...(vals) > 0)
                        {
                                next();
                        }
                        values(vals...);
                }

                // attributes
                string_t        m_text;
                const size_t    m_tabsize;      ///< todo
                const size_t    m_spacing;      ///< todo
                const bool      m_newline;      ///< todo
        };

        ///
        /// \brief limited ascii-based JSON reader.
        ///
        class json_reader_t
        {
        public:

                enum class tag
                {
                        begin_object,
                        end_object,
                        begin_array,
                        end_array,
                        token,
                        next,
                        pair,
                        null
                };

                json_reader_t(const string_t& text) :
                        m_text(text)
                {
                        skip(spaces());
                }

                template <typename tcallback>
                void parse(const tcallback& callback)
                {
                        auto prev = m_pos;
                        while (find(tokens()))
                        {
                                const auto curr = m_pos;

                                if (prev < curr)
                                {
                                        handle_token(prev, curr, callback);
                                }

                                switch (m_text[m_pos])
                                {
                                case '{':       callback(m_text, curr, ++ m_pos, tag::begin_object); break;
                                case '}':       callback(m_text, curr, ++ m_pos, tag::end_object); break;
                                case '[':       callback(m_text, curr, ++ m_pos, tag::begin_array); break;
                                case ']':       callback(m_text, curr, ++ m_pos, tag::end_array); break;
                                case ',':       callback(m_text, curr, ++ m_pos, tag::next); break;
                                case ':':       callback(m_text, curr, ++ m_pos, tag::pair); break;
                                default:        assert(false);
                                }

                                prev = ++ m_pos;
                                skip(spaces());
                        }
                }

        private:

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

                template <typename tcallback>
                void handle_token(const size_t prev, const size_t curr, const tcallback& callback) const
                {
                        auto begin = m_text.find_first_not_of(spaces(), prev);
                        auto end = std::min(curr, m_text.find_first_of(spaces(), begin));

                        if (begin < end)
                        {
                                if (m_text.compare(begin, end, "null") == 0)
                                {
                                        callback(m_text, begin, end, tag::null);
                                }
                                else
                                {
                                        if (m_text[begin] == '\"') ++ begin;
                                        if (m_text[end - 1] == '\"') --end;
                                        callback(m_text, begin, end, tag::token);
                                }
                        }
                }

                // attributes
                const string_t& m_text;
                size_t          m_pos{0};
                size_t          m_depth_object{0};
                size_t          m_depth_array{0};
        };
}
