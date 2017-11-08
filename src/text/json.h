#pragma once

#include "cast.h"

namespace nano
{
        ///
        /// \brief limited ascii-based JSON encoder.
        ///
        template <char tbegin, char tend>
        class json_encoder_t
        {
        public:

                json_encoder_t(string_t& text) : m_text(text)
                {
                        m_text += tbegin;
                }

                ~json_encoder_t()
                {
                        m_text += tend;
                }

                void name(const char* tag)
                {
                        m_text += tag;
                        m_text += ':';
                }

                template <typename tvalue>
                void pair(const char* tag, const tvalue& value)
                {
                        name(tag);
                        m_text += to_string(value);
                }

                void next() { m_text += ','; }
                void newline() { m_text += '\n'; }

                auto array(const char* tag)
                {
                        name(tag);
                        return json_encoder_t<'[', ']'>(m_text);
                }

                auto object(const char* tag)
                {
                        name(tag);
                        return json_encoder_t<'{', '}'>(m_text);
                }

        private:

                // attributes
                string_t&       m_text;
        };

        ///
        /// \brief create a JSON encoder object.
        ///
        inline auto make_json_encoder(string_t& text)
        {
                return json_encoder_t<'{', '}'>{text};
        }

        ///
        /// \brief limited ascii-based JSON decoder.
        ///
        template <char tbegin, char tend>
        class json_decoder_t
        {
        public:

                json_decoder_t(const string_t& text, const size_t pos = 0) : m_text(text), m_pos(pos)
                {
                        m_pos = m_text.find(tbegin, m_pos);
                }

                ~json_decoder_t()
                {
                        m_pos = m_text.find(tend, m_pos);
                }

        private:

                // attributes
                const string_t& m_text;
                size_t          m_pos;
        };
}
