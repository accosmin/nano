#pragma once

#include "arch.h"
#include <string>
#include <cstddef>

namespace cortex
{
        ///
        /// \brief map the std::istream's interface over an in-memory buffer
        ///
        class NANOCV_PUBLIC mstream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                mstream_t(const char* data, std::size_t size);

                ///
                /// \brief disable copying
                ///
                mstream_t(const mstream_t&) = delete;
                mstream_t& operator=(const mstream_t&) = delete;

                ///
                /// \brief read given number of bytes
                ///
                bool read(char* bytes, std::size_t max_num_bytes);

                ///
                /// \brief read POD structure
                ///
                template
                <
                        typename tstruct
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
                /// \brief skip the given number of bytes
                ///
                bool skip(std::size_t num_bytes);

                ///
                /// \brief number of bytes read at the last operation
                ///
                std::size_t gcount() const;

                ///
                /// \brief current position in the buffer
                ///
                std::size_t tellg() const;

                ///
                /// \brief buffer size
                ///
                std::size_t size() const;

                ///
                /// \brief check if EOF
                ///
                operator bool() const;

        private:

                const char* const       m_data;
                std::size_t             m_size;
                std::size_t             m_tellg;
                std::size_t             m_gcount;
        };
}
