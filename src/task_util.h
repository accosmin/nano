#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief print a short description of a task.
        ///
        NANO_PUBLIC void describe(const task_t& task);

        ///
        /// \brief save the samples of the given fold as images (if possible) to the given path.
        ///
        NANO_PUBLIC void save_as_images(const task_t&, const fold_t&, const string_t& basepath,
                const tensor_size_t grows, const tensor_size_t gcols);
}
