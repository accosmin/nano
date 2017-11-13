#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief manage sampling objects (register new ones, query and clone them)
        ///
        class enhancer_t;
        using enhancer_factory_t = factory_t<enhancer_t>;
        using renhancer_t = enhancer_factory_t::trobject;

        NANO_PUBLIC enhancer_factory_t& get_enhancers();

        ///
        /// \brief artificially augment the (training) samples, useful for improving the generalization error.
        ///
        class NANO_PUBLIC enhancer_t
        {
        public:

                virtual ~enhancer_t() = default;

                ///
                /// \brief serialize the current parameters to json
                ///
                virtual void config(json_reader_t&) = 0;
                virtual void config(json_writer_t&) const = 0;

                ///
                /// \brief retrieve the given [begin, end) samples as a minibatch
                ///
                virtual minibatch_t get(const task_t&, const fold_t&, const size_t begin, const size_t end) const = 0;
        };
}
