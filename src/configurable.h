#pragma once

#include "text/json_reader.h"
#include "text/json_writer.h"

namespace nano
{
        ///
        /// \brief interface for JSON-based configurable objects.
        ///
        class configurable_t
        {
        public:

                virtual ~configurable_t() = default;

                ///
                /// \brief deserialize and update the current parameters from JSON
                ///
                virtual json_reader_t& config(json_reader_t&) = 0;

                ///
                /// \brief serialize the current parameters to JSON
                ///
                virtual json_writer_t& config(json_writer_t&) const = 0;
        };
}
