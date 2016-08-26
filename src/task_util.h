#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief print a short description of a task.
        ///
        NANO_PUBLIC void describe(const task_t& task);

        ///
        /// \brief check task for consistency per fold index:
        ///     there should be no duplications per fold index.
        ///
        NANO_PUBLIC bool check_duplicates(const task_t& task);

        ///
        /// \brief check task for consistency per fold index:
        ///     there should be no intersection between training, validation and tests datasets.
        ///
        NANO_PUBLIC bool check_intersection(const task_t& task);

        ///
        /// \brief save the samples of the given fold as images (if possible) to the given path.
        ///
        NANO_PUBLIC void save_as_images(const task_t&, const fold_t&, const string_t& basepath,
                const tensor_size_t grows, const tensor_size_t gcols);
}
