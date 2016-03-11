#pragma once

#include "arch.h"
#include <ios>
#include <string>
#include <cstddef>

namespace nano
{
        ///
        /// \brief map the std::istream's interface over a fixed-size in-memory buffer
        ///
        class NANO_PUBLIC imstream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                template
                <
                        typename tsize
                >
                imstream_t(const char* data, const tsize size)
                        :       m_data(data),
                                m_size(static_cast<std::streamsize>(size)),
                                m_tellg(0),
                                m_gcount(0)
                {
                }

                ///
                /// \brief disable copying
                ///
                imstream_t(const imstream_t&) = delete;
                imstream_t& operator=(const imstream_t&) = delete;

                ///
                /// \brief read given number of bytes
                ///
                imstream_t& read(char* bytes, std::streamsize max_num_bytes);

                ///
                /// \brief read POD structure
                ///
                template
                <
                        typename tstruct
                >
                imstream_t& read(tstruct& pod)
                {
                        return read(reinterpret_cast<char*>(&pod), sizeof(pod));
                }

                ///
                /// \brief read next line
                ///
                imstream_t& getline(std::string& line);

                ///
                /// \brief move to the given position in the buffer
                ///
                imstream_t& seekg(std::streampos pos);

                ///
                /// \brief number of bytes read at the last operation
                ///
                std::streamsize gcount() const;

                ///
                /// \brief current position in the buffer
                ///
                std::streamsize tellg() const;

                ///
                /// \brief buffer size
                ///
                std::streamsize size() const;

                ///
                /// \brief check if EOF
                ///
                bool eof() const;

                ///
                /// \brief check state
                ///
                bool good() const;

                ///
                /// \brief check if EOF
                ///
                operator bool() const;

                ///
                /// \brief buffer data
                ///
                const char* data() const;

        private:

                const char* const       m_data;
                std::streamsize         m_size;
                std::streamsize         m_tellg;
                std::streamsize         m_gcount;
        };
}
