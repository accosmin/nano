#pragma once

#include "task_mem_vision.h"

namespace nano
{
        enum class charset_mode
        {
                digit,          ///< 0-9
                lalpha,         ///< a-z
                ualpha,         ///< A-Z
                alpha,          ///< a-zA-Z
                alphanum,       ///< A-Za-z0-9
        };

        template <>
        inline std::map<charset_mode, std::string> enum_string<charset_mode>()
        {
                return
                {
                        { charset_mode::digit,          "digit" },
                        { charset_mode::lalpha,         "lalpha" },
                        { charset_mode::ualpha,         "ualpha" },
                        { charset_mode::alpha,          "alpha" },
                        { charset_mode::alphanum,       "alphanum" }
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
