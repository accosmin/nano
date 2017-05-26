#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief manage sampling objects (register new ones, query and clone them)
        ///
        struct iterator_t;
        using iterator_factory_t = factory_t<iterator_t>;
        using riterator_t = iterator_factory_t::trobject;

        NANO_PUBLIC iterator_factory_t& get_iterators();

        ///
        /// \brief iterator (generator) over a task that
        ///     can be used for artificially augmenting the training samples,
        ///
        struct NANO_PUBLIC iterator_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief retrieve the 3D input tensor for a given sample
                ///
                virtual tensor3d_t input(const task_t& task, const fold_t& fold, const size_t index) const = 0;

                ///
                /// \brief retrieve the output target for a given sample
                ///
                virtual tensor3d_t target(const task_t& task, const fold_t& fold, const size_t index) const = 0;
        };
}
