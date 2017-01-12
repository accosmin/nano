#pragma once

#include "buffer.h"
#include <type_traits>

namespace nano
{
        ///
        /// \brief input streaming interface for binary data.
        ///
        class NANO_PUBLIC istream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                istream_t();

                ///
                /// \brief disable copying
                ///
                istream_t(const istream_t&) = delete;
                istream_t& operator=(const istream_t&) = delete;

                ///
                /// \brief enable moving
                ///
                istream_t(istream_t&&) = default;
                istream_t& operator=(istream_t&&) = default;

                ///
                /// \brief destructor
                ///
                virtual ~istream_t() = default;

                ///
                /// \brief read given number of bytes
                ///
                bool read(char* bytes, const std::streamsize max_num_bytes);

                ///
                /// \brief read POD structure
                ///
                template
                <
                        typename tstruct,
                        typename = typename std::enable_if<std::is_pod<tstruct>::value>::type
                >
                bool read(tstruct& pod)
                {
                        return read(reinterpret_cast<char*>(&pod), sizeof(pod));
                }

                ///
                /// \brief read next line
                ///
                bool getline(std::string& line);

                ///
                /// \brief current position in the buffer
                ///
                std::streamsize tellg() const;

        protected:

                enum class status
                {
                        ok,
                        eof,
                        error
                };

                virtual status advance(const std::streamsize num_bytes, buffer_t& buffer) = 0;

        private:

                buffer_t        m_buffer;       ///< current buffer
                std::streamsize m_index;        ///< position in the current buffer
                status          m_status;       ///<
                std::streamsize m_tellg;
        };
}
