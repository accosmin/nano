#pragma once

#include "io.h"
#include "arch.h"
#include <type_traits>

namespace nano
{
        ///
        /// \brief input streaming interface for binary data.
        /// NB: assuming that streaming if uni-directional (e.g. cannot move cursor to specific positions).
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
                /// \return the number of bytes actually read
                ///
                std::streamsize read(char* bytes, const std::streamsize num_bytes);

                ///
                /// \brief read POD structure
                ///
                template <typename tstruct, typename = typename std::enable_if<std::is_pod<tstruct>::value>::type>
                bool read(tstruct& pod);

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
                /// \brief number of bytes available in the buffer
                ///
                std::streamsize available() const;

                ///
                /// \brief number of bytes read at the last operation
                ///
                std::streamsize gcount() const;

                ///
                /// \brief check state
                ///
                operator bool() const;

        private:

                void trim();
                void advance();
                std::streamsize buffer(const std::streamsize num_bytes);

                virtual io_status advance(const std::streamsize num_bytes, buffer_t& buffer) = 0;

        private:

                // attributes
                buffer_t                m_buffer;       ///< buffer
                std::streamsize         m_index;        ///< position in the buffer
                io_status               m_status;       ///<
                std::streamsize         m_tellg;        ///< position since begining
                std::streamsize         m_gcount;
        };

        template <typename tstruct, typename>
        bool istream_t::read(tstruct& pod)
        {
                const auto size = static_cast<std::streamsize>(sizeof(pod));
                return read(reinterpret_cast<char*>(&pod), size) == size;
        }
}
