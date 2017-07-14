#pragma once

#include "enhancer.h"

namespace nano
{
        ///
        /// \brief generate samples by adding random samples without a label/class.
        /// NB: the idea is that natural images are just a small subset of all possible images, so most likely
        ///     a random image will have no label/class, so its target is {-1}^C (C = number of classes).
        ///
        struct enhancer_noclass_t final : public enhancer_t
        {
                explicit enhancer_noclass_t(const string_t& params = string_t());

                virtual sample_t get(const task_t&, const fold_t&, const size_t index) const final;
        };
}
