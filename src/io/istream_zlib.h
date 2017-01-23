#pragma once

#include <zlib.h>
#include <iosfwd>
#include "istream.h"

namespace nano
{
        ///
        /// \brief zlib-based streaming of gzip-compressed binary data.
        ///
        class NANO_PUBLIC zlib_istream_t final : public istream_t
        {
        public:

                zlib_istream_t(std::istream& istream, const std::streamsize max_num_bytes = max_streamsize());

                ~zlib_istream_t();

                virtual io_status advance(const std::streamsize num_bytes, buffer_t& buffer) override;

        private:

                // attributes
                std::istream&   m_istream;              ///< input stream
                std::streamsize m_max_num_bytes;        ///< maximum number of bytes to read from the input stream
                z_stream        m_zstream;              ///< zlib stream
        };
}
