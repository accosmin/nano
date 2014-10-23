#pragma once

#include "io_base.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief map the std::istream's interface over an in-memory buffer
                ///
                class stream_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        stream_t(const data_t& data);

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
                        /// \brief number of bytes to end
                        ///
                        size_t remg() const;

                        ///
                        /// \brief buffer size
                        ///
                        size_t size() const;

                        ///
                        /// \brief check if EOF
                        ///
                        operator bool() const;

                private:

                        const data_t&           m_data;
                        size_t                  m_tellg;
                        size_t                  m_gcount;
                };
        }
}
