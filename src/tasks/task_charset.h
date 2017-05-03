#pragma once

#include "task_mem_vision.h"

namespace nano
{
        enum class charset_type
        {
                digit,          ///< 0-9
                lalpha,         ///< a-z
                ualpha,         ///< A-Z
                alpha,          ///< a-zA-Z
                alphanum,       ///< A-Za-z0-9
        };

        template <>
        inline std::map<charset_type, std::string> enum_string<charset_type>()
        {
                return
                {
                        { charset_type::digit,          "digit" },
                        { charset_type::lalpha,         "lalpha" },
                        { charset_type::ualpha,         "ualpha" },
                        { charset_type::alpha,          "alpha" },
                        { charset_type::alphanum,       "alphanum" }
                };
        }

        ///
        /// \brief synthetic task to classify characters
        ///
        /// parameters:
        ///     type    - character set
        ///     irows   - sample size in pixels (rows)
        ///     icols   - sample size in pixels (columns)
        ///     color   - color mode
        ///     count   - number of samples (training + validation)
        ///
        struct NANO_PUBLIC charset_task_t final : public mem_vision_task_t
        {
                explicit charset_task_t(const string_t& configuration = string_t());

                virtual bool populate() override;
        };
}
