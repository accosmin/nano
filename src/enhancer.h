#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief manage sampling objects (register new ones, query and clone them)
        ///
        struct enhancer_t;
        using enhancer_factory_t = factory_t<enhancer_t>;
        using renhancer_t = enhancer_factory_t::trobject;

        NANO_PUBLIC enhancer_factory_t& get_enhancers();

        ///
        /// \brief artificially augment the (training) samples, useful for improving the generalization error.
        ///
        struct NANO_PUBLIC enhancer_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief retrieve the given sample
                ///
                virtual sample_t get(const task_t&, const fold_t&, const size_t index) const = 0;

                ///
                /// \brief retrieve the given [begin, end) samples as a minibatch
                ///
                virtual minibatch_t get(const task_t&, const fold_t&, const size_t begin, const size_t end) const = 0;
        };
}
