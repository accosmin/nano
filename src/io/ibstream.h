#pragma once

#include "arch.h"
#include <iosfwd>
#include <cstddef>
#include <type_traits>

namespace io
{
        ///
        /// \brief wrapper over binary std::istream
        ///
        class NANOCV_PUBLIC ibstream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                ibstream_t(std::istream& os);

                ///
                /// \brief read a POD structure
                ///
                template
                <
                        typename tstruct,
                        typename = typename std::enable_if<std::is_pod<tstruct>::value>::type
                >
                ibstream_t& read(tstruct& pod)
                {
                        return read_blob(reinterpret_cast<char*>(&pod), sizeof(pod));
                }

                ///
                /// \brief read a string
                ///
                ibstream_t& read(std::string& str);

                ///
                /// \brief read an array of the given size
                ///
                template
                <
                        typename tvalue,
                        typename tsize
                >
                ibstream_t& read(tvalue* data, const tsize count)
                {
                        return read_blob(reinterpret_cast<char*>(data),
                                          static_cast<std::size_t>(count) * sizeof(tvalue));
                }

        private:

                ibstream_t& read_blob(char* data, const std::size_t count);

        private:

                // attributes
                std::istream&   m_stream;
        };
}
