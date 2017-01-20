#pragma once

#include "buffer.h"
#include <algorithm>
#include <type_traits>

namespace nano
{
        ///
        /// \brief input streaming interface for binary data.
        ///
        template <typename treader>
        class istream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                istream_t(treader& reader);

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
                ~istream_t() = default;

                ///
                /// \brief read given number of bytes
                ///
                bool read(char* bytes, const std::streamsize num_bytes);

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
                /// \brief skip the given number of bytes
                ///
                bool skip(const std::streamsize num_bytes);

                ///
                /// \brief skip till the end and returns tellg()
                ///
                std::streamsize skip();

                ///
                /// \brief read next line
                ///
                bool getline(std::string& line);

                ///
                /// \brief current position in the (uncompressed/decoded) stream
                ///
                std::streamsize tellg() const;

                ///
                /// \brief available number of bytes
                ///
                std::streamsize gcount() const;

        private:

                static bool isendl(char c)
                {
                        return (c == '\n') || (c == '\r');
                }

                static size_t max_buffer_size()
                {
                        return size_t(1024) * size_t(1024);
                }

                void trim();
                void advance(const std::streamsize num_bytes);
                std::streamsize buffer(const std::streamsize num_bytes);

        private:

                // attributes
                treader&                m_reader;
                buffer_t                m_buffer;       ///< buffer
                std::streamsize         m_index;        ///< position in the buffer
                io_status               m_status;       ///<
                std::streamsize         m_tellg;        ///< position since begining
        };

        template <typename treader>
        istream_t<treader>::istream_t(treader& reader) :
                m_reader(reader),
                m_index(0),
                m_status(io_status::ok),
                m_tellg(0)
        {
        }

        template <typename treader>
        std::streamsize istream_t<treader>::buffer(const std::streamsize num_bytes)
        {
                // read the missing buffered data (if possible)
                if (    m_status == io_status::ok &&
                        m_index + num_bytes > static_cast<std::streamsize>(m_buffer.size()))
                {
                        m_status = m_reader.advance(m_index + num_bytes, m_buffer);
                }

                // return the number of bytes available
                return gcount();
        }

        template <typename treader>
        void istream_t<treader>::advance(const std::streamsize num_bytes)
        {
                m_index += num_bytes;
                m_tellg += num_bytes;
        }

        template <typename treader>
        void istream_t<treader>::trim()
        {
                // keep buffer small enough
                if (m_buffer.size() > max_buffer_size())
                {
                        m_buffer.erase(m_buffer.begin(), m_buffer.begin() + m_index);
                        m_index = 0;
                }
        }

        template <typename treader>
        bool istream_t<treader>::read(char* bytes, const std::streamsize num_bytes)
        {
                if (buffer(num_bytes) >= num_bytes && m_status != io_status::error)
                {
                        const char* data = m_buffer.data();
                        std::copy(data + m_index, data + (m_index + num_bytes), bytes);
                        advance(num_bytes);
                        trim();
                        return true;
                }
                else
                {
                        return false;
                }
        }

        template <typename treader>
        std::streamsize istream_t<treader>::skip()
        {
                const auto num_bytes = std::streamsize(64 * 1024);
                auto read_bytes = std::streamsize(0);
                while ((read_bytes = buffer(num_bytes)) > 0 && m_status == io_status::ok)
                {
                        advance(read_bytes);
                        trim();
                }
                return tellg();
        }

        template <typename treader>
        bool istream_t<treader>::getline(std::string& line)
        {
                /// \todo not very efficient: should buffer larger chunks (1K ?!) and check for endline there!
                char c;
                while (read(&c, 1) && isendl(c)) {}

                line.clear();
                while (read(&c, 1) && !isendl(c)) { line.push_back(c); }

                return m_status != io_status::error && !line.empty();
        }

        template <typename treader>
        std::streamsize istream_t<treader>::tellg() const
        {
                return m_tellg;
        }

        template <typename treader>
        std::streamsize istream_t<treader>::gcount() const
        {
                return static_cast<std::streamsize>(m_buffer.size()) - m_index;
        }
}
