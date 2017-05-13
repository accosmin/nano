#pragma once

#include "task_mem_vision.h"

namespace nano
{
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
        struct charset_task_t final : public mem_vision_task_t
        {
                explicit charset_task_t(const string_t& configuration = string_t());

                virtual bool populate() override;
        };
}
