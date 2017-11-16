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

                json_writer_t& null()
                {
                        m_text += "null";
                        return *this;
                }

                template <typename... tvalues>
                json_writer_t& array(const tvalues&... vals)
                {
                        new_array();
                        values(vals...);
                        return end_array();
                }

                json_writer_t& pairs()
                {
                        return *this;
                }

                template <typename tvalue, typename... tvalues>
                json_writer_t& pairs(const char* tag, const tvalue& val, const tvalues&... vals)
                {
                        pair(tag, val);
                        if (sizeof...(vals))
                        {
                                next();
                        }
                        return pairs(vals...);
                }

                template <typename... tvalues>
                json_writer_t& object(const tvalues&... vals)
                {
                        new_object();
                        pairs(vals...);
                        return end_object();
                }

                json_writer_t& new_array() { return keyword('['); }
                json_writer_t& end_array() { return keyword(']'); }

                json_writer_t& new_object() { return keyword('{'); }
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
        };
}
