#pragma once

#include "text/json.h"

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
                /// \brief serialize to JSON
                ///
                virtual void to_json(json_t&) const = 0;
                virtual void from_json(const json_t&) = 0;
        };
}
