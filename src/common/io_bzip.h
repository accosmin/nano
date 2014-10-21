#pragma once

#include "io.h"

namespace ncv
{
        namespace io
        {
                ///
                /// \brief uncompress a stream of bytes (using bzip2)
                ///
                bool uncompress_bzip2(std::istream& istream, std::size_t bytes, data_t& data);
                bool uncompress_bzip2(std::istream& istream, data_t& data);

                bool uncompress_bzip2(std::istream& istream, std::size_t bytes, const data_callback_t& callback);
                bool uncompress_bzip2(std::istream& istream, const data_callback_t& callback);

                bool uncompress_bzip2(std::istream& istream, std::size_t bytes, const data_callback_t& callback);
                bool uncompress_bzip2(std::istream& istream, const data_callback_t& callback);
        }
}
