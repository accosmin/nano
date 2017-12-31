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

                json_writer_t& name(const char* tag)
                {
                        return quote(tag).append(':');
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
                                return append(to_string(val));
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
                        return name(tag).value(val);
                }

                json_writer_t& next()
                {
                        return append(',');
                }

                json_writer_t& null()
                {
                        return append("null");
                }

                template <typename... tvalues>
                json_writer_t& array(const tvalues&... vals)
                {
                        return new_array().values(vals...).end_array();
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
                        return new_object().pairs(vals...).end_object();
                }

                template <typename tstr>
                json_writer_t& append(const tstr& str)
                {
                        m_str += str;
                        return *this;
                }

                json_writer_t& new_array() { return append('['); }
                json_writer_t& end_array() { return append(']'); }

                json_writer_t& new_object() { return append('{'); }
                json_writer_t& end_object() { return append('}'); }

                ///
                /// \brief returns the current JSON string
                ///
                const auto& str() { return m_str; }

        private:

                template <typename tstr>
                json_writer_t& quote(const tstr& str)
                {
                        return append('\"').append(str).append('\"');
                }

                template <typename tvalue>
                json_writer_t& values(const tvalue& val)
                {
                        value(val);
                        return *this;
                }

                template <typename tvalue, typename... tvalues>
                json_writer_t& values(const tvalue& val, const tvalues&... vals)
                {
                        value(val);
                        if (sizeof...(vals) > 0)
                        {
                                next();
                        }
                        values(vals...);
                        return *this;
                }

                // attributes
                string_t        m_str;          ///< current JSON string
        };
}
