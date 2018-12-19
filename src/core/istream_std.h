#pragma once

#include <iosfwd>
#include "istream.h"

namespace nano
{
        ///
        /// \brief streaming of binary data using a std::istream.
        ///
        // cppcheck-suppress class_X_Y
        class NANO_PUBLIC std_istream_t final : public istream_t
        {
        public:
                explicit std_istream_t(std::istream& stream);

                io_status advance(const std::streamsize num_bytes, buffer_t& buffer) override;

        private:

                std::istream&           m_stream;
        };
}
