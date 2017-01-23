#pragma once

#include "istream.h"

namespace nano
{
        ///
        /// \brief streaming of binary data using an in-memory buffer.
        ///
        class NANO_PUBLIC mem_istream_t final : public istream_t
        {
        public:

                template <typename tsize>
                mem_istream_t(const char* data, const tsize size) :
                        m_data(data),
                        m_size(static_cast<std::streamsize>(size)),
                        m_index(0)
                {
                }

                ~mem_istream_t() = default;

                virtual io_status advance(const std::streamsize num_bytes, buffer_t& buffer) override;

        private:

                const char* const       m_data;
                const std::streamsize   m_size;
                std::streamsize         m_index;
        };
}
