#pragma once

#include "iterator.h"

namespace nano
{
        ///
        /// \brief generate samples by adding random samples without a label/class.
        /// NB: the idea is that natural images are just a small subset of all possible images, so most likely
        ///     a random image will have no label/class, so its target is {-1}^C (C = number of classes).
        ///
        struct iterator_noclass_t final : public iterator_t
        {
                explicit iterator_noclass_t(const string_t& configuration = string_t());

                virtual sample_t get(const task_t&, const fold_t&, const size_t index) const final;
        };
}
