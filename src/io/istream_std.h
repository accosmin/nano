#pragma once

#include <iosfwd>
#include "istream.h"

namespace nano
{
        ///
        /// \brief streaming of binary data using a std::istream.
        ///
        struct NANO_PUBLIC std_istream_t final : public istream_t
        {
                explicit std_istream_t(std::istream& stream);

                ~std_istream_t() override = default;

                io_status advance(const std::streamsize num_bytes, buffer_t& buffer) override;

        private:

                std::istream&           m_stream;
        };
}
