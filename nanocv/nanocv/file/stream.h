#pragma once

#include <string>
#include <utility>

namespace ncv
{
        namespace io
        {
                using std::size_t;
                
                ///
                /// \brief map the std::istream's interface over an in-memory buffer
                ///
                class stream_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        stream_t(const char* data, size_t size);

                        ///
                        /// \brief disable copying
                        ///
                        stream_t(const stream_t&) = delete;
                        stream_t& operator=(const stream_t&) = delete;

                        ///
                        /// \brief read given number of bytes
                        ///
                        bool read(char* bytes, size_t max_num_bytes);

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
                        bool skip(size_t num_bytes);

                        ///
                        /// \brief number of bytes read at the last operation
                        ///
                        size_t gcount() const;

                        ///
                        /// \brief current position in the buffer
                        ///
                        size_t tellg() const;

                        ///
                        /// \brief buffer size
                        ///
                        size_t size() const;

                        ///
                        /// \brief check if EOF
                        ///
                        operator bool() const;

                private:

                        const char* const       m_data;
                        size_t                  m_size;
                        size_t                  m_tellg;
                        size_t                  m_gcount;
                };
        }
}
