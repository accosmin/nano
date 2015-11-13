#pragma once

#include "arch.h"
#include <iosfwd>
#include <string>
#include <cstddef>
#include <type_traits>

namespace file
{
        ///
        /// \brief wrapper over binary std::ostream
        ///
        class NANOCV_PUBLIC obstream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                obstream_t(std::ostream& os);

                ///
                /// \brief write a POD structure
                ///
                template
                <
                        typename tstruct,
                        typename = typename std::enable_if<std::is_pod<tstruct>::value>::type
                >
                obstream_t& write(const tstruct& pod)
                {
                        return write_blob(reinterpret_cast<const char*>(&pod),
                                          sizeof(pod));
                }

                ///
                /// \brief write a string
                ///
                obstream_t& write(const std::string& str);

                ///
                /// \brief write an array of the given size
                ///
                template
                <
                        typename tvalue,
                        typename tsize
                >
                obstream_t& write(const tvalue* data, const tsize count)
                {
                        return write_blob(reinterpret_cast<const char*>(data),
                                          static_cast<std::size_t>(count) * sizeof(tvalue));
                }

        private:

                obstream_t& write_blob(const char* data, const std::size_t count);

        private:

                // attributes
                std::ostream&   m_stream;
        };
}
