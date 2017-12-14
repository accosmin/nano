#pragma once

#include "cast.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief limited ascii-based JSON writer.
        ///
        class json_writer_t
        {
        public:

                enum class format
                {
                        compact,
                        humanx4
                };

                ///
                /// \brief constructor
                ///
                json_writer_t(const format fmt = format::compact) : m_format(fmt)
                {
                }

                json_writer_t& name(const char* tag)
                {
                        quote(tag);
                        m_str += ':';
                        return *this;
                }

                template <typename tvalue>
                json_writer_t& value(const tvalue& val)
                {
                        // todo: use if constexpr instead when moving to c++17
                        if (std::is_enum<tvalue>::value)
                        {
                                return quote(to_string(val));
                        }
                        else
                        {
                                m_str.append(to_string(val));
                                return *this;
                        }
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
                        return prefix().name(tag).value(val);
                }

                json_writer_t& next()
                {
                        m_str.append(1, ',');
                        return newline();
                }

                json_writer_t& null()
                {
                        m_str += "null";
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

                json_writer_t& new_array() { return keyword('[').newline(); }
                json_writer_t& end_array() { return keyword(']').newline(); }

                json_writer_t& new_object() { return newline().prefix().keyword('{').newline().increase_prefix(); }
                json_writer_t& end_object() { return newline().keyword('}').decrease_prefix(); }

                ///
                /// \brief returns the current JSON string
                ///
                const auto& str() { return m_str; }

        private:

                json_writer_t& newline()
                {
                        switch (m_format)
                        {
                        case format::compact:   return *this;
                        default:                return keyword('\n');
                        }
                }

                json_writer_t& prefix()
                {
                        switch (m_format)
                        {
                        case format::compact:   return *this;
                        default:                m_str += m_prefix; return *this;
                        }
                }

                json_writer_t& increase_prefix()
                {
                        switch (m_format)
                        {
                        case format::compact:   return *this;
                        case format::humanx4:   m_prefix += "    "; return *this;
                        default:                assert(false); return *this;
                        }
                }

                json_writer_t& decrease_prefix()
                {
                        switch (m_format)
                        {
                        case format::compact:   return *this;
                        case format::humanx4:   m_prefix.erase(m_prefix.size() - 4, 4); return *this;
                        default:                assert(false); return *this;
                        }
                }

                json_writer_t& keyword(const char tag)
                {
                        m_str += tag;
                        return *this;
                }

                template <typename tstr>
                json_writer_t& quote(const tstr& str)
                {
                        m_str += '\"';
                        m_str += str;
                        m_str += '\"';
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
                string_t        m_str;          ///< current JSON string
                format          m_format;       ///< formatting options
                string_t        m_prefix;       ///< current prefix given
        };
}
